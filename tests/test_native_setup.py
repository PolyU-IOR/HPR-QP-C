"""Native library reuse, download opt-out and verified private installation."""
import importlib.util
import io
import hashlib
import ssl
import urllib.error
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec=importlib.util.spec_from_file_location('native_setup',Path(__file__).resolve().parents[1]/'scripts/setup_native.py')
setup=importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

class NativeSetup(unittest.TestCase):
    def test_apt_requires_strong_hash_and_plain_filename(self):
        row="'http://mirror/native.deb' native.deb 42 SHA256:"+'a'*64
        self.assertEqual(setup.package_records(row)[0]['size'],42)
        for bad in (row.replace('SHA256','MD5Sum'),row.replace("' native.deb","' ../native.deb"),'no metadata'):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                setup.package_records(bad)

    def test_download_verifies_checksum(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'open_download',return_value=io.BytesIO(b'payload')):
            p=Path(tmp)/'native.deb'
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                setup.download('https://mirror/native.deb',p,'0'*64)

    def test_bad_download_checksum_cannot_publish_native_install(self):
        rows='\n'.join("'http://mirror/p%d.deb' p%d.deb 1 SHA256:%s"%(i,i,'a'*64) for i in range(5))
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'supported'), \
                patch.object(setup.shutil,'which',return_value='/tool'), \
                patch.object(setup.subprocess,'check_output',side_effect=['',rows]), \
                patch.object(setup,'download',side_effect=RuntimeError('checksum mismatch')):
            p=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                setup.install_libraries(p)
            self.assertEqual(list(p.iterdir()),[])

    def test_existing_private_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'supported'), \
                patch.object(setup.shutil,'which',return_value='/tool'), \
                patch.object(setup,'download') as download:
            p=Path(tmp);(p/'native').mkdir();keep=p/'native/keep';keep.write_text('existing')
            with self.assertRaisesRegex(RuntimeError,'nothing overwritten'):
                setup.install_libraries(p)
            self.assertEqual(keep.read_text(),'existing');download.assert_not_called()

    def test_existing_native_libraries_skip_install(self):
        libraries=dict(blas='/system/blas',lapack='/system/lapack')
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'ROOT',Path(tmp)), \
                patch.object(setup,'discover_libraries',return_value=libraries), \
                patch.object(setup,'install_libraries') as install:
            self.assertEqual(setup.setup(Path(tmp)/'build',False),libraries)
            install.assert_not_called()

    def test_missing_libraries_install_once_then_verify_without_cmake(self):
        libraries=dict(blas='/private/blas',lapack='/private/lapack')
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'ROOT',Path(tmp)), \
                patch.object(setup,'discover_libraries',side_effect=[None,libraries]), \
                patch.object(setup,'install_libraries') as install:
            self.assertEqual(setup.setup(Path(tmp)/'build',True),libraries)
            install.assert_called_once_with(Path(tmp)/'.deps',allow_download=True)

    def test_failed_post_install_probe_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'ROOT',Path(tmp)), \
                patch.object(setup,'discover_libraries',return_value=None), \
                patch.object(setup,'install_libraries') as install:
            with self.assertRaisesRegex(RuntimeError,'cannot link or start'):
                setup.setup(Path(tmp)/'build',True)
            install.assert_called_once()

    def test_download_opt_out_does_not_access_apt_or_network(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(setup,'supported'), \
                patch.object(setup,'download') as download, \
                patch.object(setup.subprocess,'check_output') as apt:
            with self.assertRaisesRegex(RuntimeError,'AUTO_INSTALL_NATIVE_DEPS=0'):
                setup.install_libraries(Path(tmp),allow_download=False)
            download.assert_not_called();apt.assert_not_called()


class PackageTransport(unittest.TestCase):
    url = 'http://hk.archive.ubuntu.com/ubuntu/pool/main/l/lapack/native.deb'
    official = 'https://archive.ubuntu.com/ubuntu/pool/main/l/lapack/native.deb'

    def test_official_https_preserves_exact_package_path(self):
        self.assertEqual(setup.native_package_urls(self.url), [self.official,
            self.url.replace('http:', 'https:'), self.url])
        self.assertEqual(setup.native_package_urls(self.official), [self.official])
        security = self.url.replace('hk.archive', 'security')
        self.assertEqual(setup.native_package_urls(security)[0], security.replace('http:', 'https:'))

    def test_custom_or_signed_urls_are_not_rewritten(self):
        for url in ('http://mirror.example/ubuntu/pool/native.deb',
                    self.url.replace('/ubuntu/pool/', '/other/'),
                    self.url + '?signature=abc', self.url + '#fragment',
                    self.url.replace('hk.archive.ubuntu.com', 'hk.archive.ubuntu.com:8080'),
                    self.url.replace('hk.archive.ubuntu.com', 'user:secret@hk.archive.ubuntu.com'),
                    self.url.replace('hk.archive.ubuntu.com', '@hk.archive.ubuntu.com'),
                    self.url.replace('hk.archive.ubuntu.com', ':@hk.archive.ubuntu.com'),
                    self.url.replace('hk.archive.ubuntu.com', 'hk.archive.ubuntu.com.evil.example')):
            with self.subTest(url=url):
                self.assertEqual(setup.native_package_urls(url), [url])

    def test_reachable_official_mirror_avoids_failed_regional_transport(self):
        payload = b'verified native package'
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'open_download', return_value=io.BytesIO(payload)) as opener:
            p = Path(tmp)/'native.deb'
            actual = setup.download(self.url, p, hashlib.sha256(payload).hexdigest())
            self.assertEqual(actual, self.official)
            self.assertEqual(p.read_bytes(), payload)
            opener.assert_called_once_with(self.official, timeout=30)

    def test_transport_failure_can_try_next_identical_package(self):
        payload = b'verified native package'
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'open_download', side_effect=[
                    urllib.error.URLError('unreachable'), io.BytesIO(payload)]) as opener:
            actual = setup.download(self.url, Path(tmp)/'native.deb', hashlib.sha512(payload).hexdigest(), 'sha512')
            self.assertEqual(actual, self.url.replace('http:', 'https:'))
            self.assertEqual(opener.call_count, 2)

    def test_checksum_mismatch_stops_without_mirror_retry(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'open_download', return_value=io.BytesIO(b'corrupt')) as opener:
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                setup.download(self.url, Path(tmp)/'native.deb', '0'*64)
            self.assertEqual(opener.call_count, 1)

    def test_tls_failures_are_not_retried(self):
        for error in (RuntimeError('Dependency download failed TLS certificate verification'),
                      ssl.SSLCertVerificationError('invalid CA'),
                      urllib.error.URLError(ssl.SSLCertVerificationError('invalid CA'))):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(setup, 'open_download', side_effect=error) as opener:
                with self.assertRaises((RuntimeError, ssl.SSLError, urllib.error.URLError)):
                    setup.download(self.url, Path(tmp)/'native.deb', '0'*64)
                self.assertEqual(opener.call_count, 1)

    def test_disk_write_failure_is_fatal_without_mirror_retry(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'open_download', return_value=io.BytesIO(b'payload')) as opener, \
                patch.object(Path, 'open', side_effect=PermissionError('disk write denied')):
            with self.assertRaises(PermissionError):
                setup.download(self.url, Path(tmp)/'native.deb', '0'*64)
            self.assertEqual(opener.call_count, 1)

    def test_failed_transports_identify_urls_and_remove_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'open_download', side_effect=urllib.error.URLError('unreachable')) as opener:
            p = Path(tmp)/'native.deb'
            p.write_bytes(b'partial earlier response')
            with self.assertRaisesRegex(RuntimeError, 'Attempted package URLs') as caught:
                setup.download(self.url, p, '0'*64)
            self.assertIn(self.official, str(caught.exception))
            self.assertIn(self.url, str(caught.exception))
            self.assertFalse(p.exists())
            self.assertEqual(opener.call_count, 3)


if __name__=='__main__':unittest.main()
