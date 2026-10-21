"""Native offline bundle, host eligibility and private-install integrity contracts."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('native_setup', Path(__file__).resolve().parents[1] / 'scripts/setup_native.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

REQUIRED = ('libblas.so.3', 'liblapack.so.3', 'libgfortran.so.5', 'libquadmath.so.0', 'libgcc_s.so.1')
PACKAGES = ('libblas3', 'liblapack3', 'libgfortran5', 'libquadmath0', 'libgcc-s1')
def sha(data):
    return hashlib.sha256(data).hexdigest()

class NativeSetup(unittest.TestCase):
    def fixture(self, root, unsafe=None, wrong_file_hash=False):
        bundle = root / 'deps/native'
        bundle.mkdir(parents=True)
        files = {'lib/' + n: n.encode() for n in REQUIRED}
        files.update({'licenses/' + n + '/copyright': b'original notice' for n in PACKAGES})
        files.update({'SOURCES.md': b'source locations', 'ORIGIN_INSTALL.json': b'{}'})
        if unsafe:
            files[unsafe] = b'escape'
        archive = bundle / 'runtime.tar.xz'
        with tarfile.open(archive, 'w:xz') as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                if unsafe == name:
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/tmp/outside'
                    info.size = 0
                    tf.addfile(info)
                else:
                    tf.addfile(info, io.BytesIO(data))
        manifest = dict(format=1, platform=dict(system='Linux', machine='x86_64', os_id='ubuntu', version_id='22.04'),
                        archive=archive.name, sha256=sha(archive.read_bytes()), size=archive.stat().st_size,
                        unpacked_bytes=sum(map(len, files.values())), packages=[],
                        files={n: dict(sha256=sha(d), size=len(d)) for n, d in files.items()},
                        libraries={n: sha(files['lib/' + n]) for n in REQUIRED})
        if wrong_file_hash:
            manifest['files']['lib/libblas.so.3']['sha256'] = '0' * 64
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
        deps = root / '.deps'
        deps.mkdir()
        return deps, archive

    def matching(self, root):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(setup, 'ROOT', root))
        stack.enter_context(patch.object(setup.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(setup.platform, 'machine', return_value='x86_64'))
        stack.enter_context(patch.object(setup, 'host_distribution', return_value=dict(ID='ubuntu', VERSION_ID='22.04')))
        return stack

    def test_matching_bundle_installs_offline_without_apt_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps, _ = self.fixture(root)
            with self.matching(root), patch.object(setup, 'download', side_effect=AssertionError('network')), \
                    patch.object(setup.subprocess, 'check_output', side_effect=AssertionError('APT')):
                setup.install_libraries(deps, allow_download=False)
            receipt = json.loads((deps / 'native/HPRQP_INSTALL.json').read_text())
            self.assertEqual(receipt['origin'], 'bundled')
            self.assertEqual(set(receipt['libraries']), set(REQUIRED))
            self.assertTrue((deps / 'native/licenses/liblapack3/copyright').is_file())
            self.assertTrue((deps / 'native/SOURCES.md').is_file())

    def test_bundle_declines_other_distro_version_and_arch(self):
        for changed in (dict(ID='debian', VERSION_ID='22.04'), dict(ID='ubuntu', VERSION_ID='24.04')):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                deps, _ = self.fixture(root)
                with self.matching(root), patch.object(setup, 'host_distribution', return_value=changed):
                    self.assertFalse(setup.install_bundled_libraries(deps))
                self.assertFalse((deps / 'native').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps, _ = self.fixture(root)
            with self.matching(root), patch.object(setup.platform, 'machine', return_value='aarch64'):
                self.assertFalse(setup.install_bundled_libraries(deps))

    def test_corrupt_bundle_never_publishes_or_falls_back_to_network(self):
        for mode in ('archive', 'file', 'missing_notice'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                deps, archive = self.fixture(root, wrong_file_hash=mode == 'file')
                if mode == 'archive':
                    archive.write_bytes(archive.read_bytes() + b'corrupt')
                if mode == 'missing_notice':
                    p = root / 'deps/native/manifest.json'
                    m = json.loads(p.read_text())
                    del m['files']['licenses/libblas3/copyright']
                    p.write_text(json.dumps(m))
                with self.matching(root), patch.object(setup, 'download', side_effect=AssertionError('network')):
                    with self.assertRaises(RuntimeError):
                        setup.install_libraries(deps)
                self.assertEqual(list(deps.iterdir()), [])

    def test_archive_rejects_escape_and_symlink(self):
        for unsafe in ('../outside', '/absolute', 'lib/link'):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                deps, _ = self.fixture(root, unsafe=unsafe)
                with self.matching(root), self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                    setup.install_bundled_libraries(deps)
                self.assertEqual(list(deps.iterdir()), [])

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps, _ = self.fixture(root)
            (deps / 'native').mkdir()
            keep = deps / 'native/keep'
            keep.write_text('existing')
            with self.matching(root), self.assertRaisesRegex(RuntimeError, 'nothing overwritten'):
                setup.install_bundled_libraries(deps)
            self.assertEqual(keep.read_text(), 'existing')

    def test_apt_requires_strong_hash_and_plain_filename(self):
        row = "'http://mirror/native.deb' native.deb 42 SHA256:" + 'a' * 64
        self.assertEqual(setup.package_records(row)[0]['size'], 42)
        for bad in (row.replace('SHA256', 'MD5Sum'), row.replace("' native.deb", "' ../native.deb"), 'no metadata'):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                setup.package_records(bad)

    def test_bad_download_checksum_cannot_publish_native_install(self):
        rows = '\n'.join("'http://mirror/p%d.deb' p%d.deb 1 SHA256:%s" % (i, i, 'a' * 64) for i in range(5))
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'supported'), \
                patch.object(setup, 'install_bundled_libraries', return_value=False), \
                patch.object(setup.shutil, 'which', return_value='/tool'), \
                patch.object(setup.subprocess, 'check_output', side_effect=['', rows]), \
                patch.object(setup, 'download', side_effect=RuntimeError('checksum mismatch')):
            p = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                setup.install_libraries(p)
            self.assertEqual(list(p.iterdir()), [])

    def test_existing_native_libraries_skip_install(self):
        libraries = dict(blas='/system/blas', lapack='/system/lapack')
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'ROOT', Path(tmp)), \
                patch.object(setup, 'discover_libraries', return_value=libraries), \
                patch.object(setup, 'install_libraries') as install:
            self.assertEqual(setup.setup(Path(tmp) / 'build', False), libraries)
            install.assert_not_called()

    def test_missing_libraries_install_once_then_verify_without_cmake(self):
        libraries = dict(blas='/private/blas', lapack='/private/lapack')
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'ROOT', Path(tmp)), \
                patch.object(setup, 'discover_libraries', side_effect=[None, libraries]), \
                patch.object(setup, 'install_libraries') as install:
            self.assertEqual(setup.setup(Path(tmp) / 'build', False), libraries)
            install.assert_called_once_with(Path(tmp) / '.deps', allow_download=False)

    def test_offline_nonmatching_host_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'supported'), \
                patch.object(setup, 'install_bundled_libraries', return_value=False), \
                patch.object(setup, 'download') as download:
            with self.assertRaisesRegex(RuntimeError, 'AUTO_INSTALL_NATIVE_DEPS=0'):
                setup.install_libraries(Path(tmp), allow_download=False)
            download.assert_not_called()

    def test_shipped_payload_and_notices_match_manifest(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp, self.matching(root):
            self.assertTrue(setup.install_bundled_libraries(Path(tmp)))
            licenses = Path(tmp) / 'native/licenses'
            self.assertTrue((licenses / 'common-licenses/GPL-3').is_file())
            self.assertIn('Runtime Library Exception', (licenses / 'libgfortran5/copyright').read_text())

if __name__ == '__main__':
    unittest.main()
