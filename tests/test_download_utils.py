"""CA discovery and fail-closed TLS contracts; no external network."""
import io
import os
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_utils as download
import setup_cudss
import setup_native


class DownloadTrust(unittest.TestCase):
    def test_default_context_always_verifies_hostname_and_chain(self):
        with patch.dict(os.environ, {}, clear=True):
            context = download.download_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_distribution_store_repairs_custom_openssl_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "system.pem"
            bundle.write_text("mock CA")
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(download.ssl, "create_default_context") as factory, \
                    patch.object(download, "SYSTEM_CA_FILES", [str(bundle)]):
                download.download_context()
                factory.return_value.load_verify_locations.assert_called_once_with(cafile=str(bundle))

    def test_explicit_trust_configuration_is_never_replaced(self):
        for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            for value in ("/explicit/trust", ""):
                with self.subTest(variable=variable, value=value), \
                        patch.dict(os.environ, {variable: value}, clear=True), \
                        patch.object(download.ssl, "create_default_context") as factory:
                    download.download_context()
                    factory.return_value.load_verify_locations.assert_not_called()

    def test_certificate_failure_is_fatal_without_unverified_retry(self):
        error = urllib.error.URLError(ssl.SSLCertVerificationError(1, "untrusted issuer"))
        with patch.object(download.urllib.request, "urlopen", side_effect=error) as fetch:
            with self.assertRaisesRegex(RuntimeError, "SSL_CERT_FILE"):
                download.open_download("https://untrusted.invalid/archive")
        self.assertEqual(fetch.call_count, 1)
        context = fetch.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_non_certificate_errors_are_not_hidden(self):
        error = urllib.error.URLError("connection refused")
        with patch.object(download.urllib.request, "urlopen", side_effect=error) as fetch:
            with self.assertRaises(urllib.error.URLError):
                download.open_download("https://unreachable.invalid/archive")
        self.assertEqual(fetch.call_count, 1)

    def test_installers_use_verified_opener_and_keep_checksum_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "archive"
            with patch.object(setup_cudss, "open_download", return_value=io.BytesIO(b"test")):
                digest = setup_cudss.download("https://test.invalid/archive", destination)
            self.assertEqual(digest, setup_native.digest(destination))
            with patch.object(setup_native, "open_download", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    setup_native.download("https://test.invalid/archive", destination, digest)


if __name__ == "__main__":
    unittest.main()
