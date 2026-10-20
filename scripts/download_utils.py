"""Verified dependency downloads using the host's configured or system CA store."""
import os
from pathlib import Path
import ssl
import urllib.error
import urllib.request

# Some custom OpenSSL builds point Python at an empty installation-prefix
# trust store. These are distribution-managed CA bundles, not downloaded roots.
SYSTEM_CA_FILES = (
    "/etc/ssl/certs/ca-certificates.crt",             # Debian/Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",               # RHEL/Fedora
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/ssl/ca-bundle.pem",                        # SUSE
    "/etc/ssl/cert.pem",                            # Alpine/BSD
)


def download_context():
    context = ssl.create_default_context()
    # An explicit trust configuration remains authoritative, including an
    # invalid/missing path. Never silently substitute another store for it.
    if "SSL_CERT_FILE" not in os.environ and "SSL_CERT_DIR" not in os.environ:
        for filename in SYSTEM_CA_FILES:
            if Path(filename).is_file():
                context.load_verify_locations(cafile=filename)
                break
    return context


def open_download(url, timeout=60):
    try:
        return urllib.request.urlopen(url, timeout=timeout, context=download_context())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "Dependency download failed TLS certificate verification. "
                "Install/update your system ca-certificates package, or set "
                "SSL_CERT_FILE to your administrator-provided trusted CA bundle "
                "and rerun make. TLS verification has not been disabled."
            ) from exc
        raise
