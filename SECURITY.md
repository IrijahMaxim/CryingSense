# Security Policy

## Supported Versions

The CryingSense project is committed to maintaining security standards for all its dependencies.

| Component | Version | Status |
|-----------|---------|--------|
| PyTorch   | >= 2.6.0 | ✅ Supported |
| PyTorch   | < 2.6.0  | ⚠️ Vulnerable |

## Security Updates

### PyTorch Security Vulnerabilities (Addressed)

**Date**: 2024-02-04

The following vulnerabilities were identified and addressed by upgrading PyTorch from 2.1.0 to 2.6.0+:

1. **Heap Buffer Overflow Vulnerability**
   - Affected versions: < 2.2.0
   - Patched in: 2.2.0
   - Severity: High

2. **Use-After-Free Vulnerability**
   - Affected versions: < 2.2.0
   - Patched in: 2.2.0
   - Severity: High

3. **Remote Code Execution via torch.load**
   - Affected versions: < 2.6.0
   - Patched in: 2.6.0
   - Severity: Critical
   - Note: When using `torch.load`, always use `weights_only=True` for untrusted sources

4. **Deserialization Vulnerability**
   - Affected versions: <= 2.3.1
   - Status: Withdrawn advisory
   - Mitigation: Use PyTorch 2.6.0+ and follow best practices for model loading

## Best Practices

### Model Loading
When loading PyTorch models, especially from untrusted sources:

```python
# Recommended: Use weights_only=True for untrusted models
model = torch.load('model.pth', weights_only=True)

# Alternative: Use torch.load with pickle restrictions
import torch
torch.serialization.add_safe_globals([YourModelClass])
model = torch.load('model.pth')
```

### Dependency Updates
- Regularly update dependencies to the latest stable versions
- Monitor security advisories for PyTorch and other ML frameworks
- Use dependency scanning tools in CI/CD pipeline

## Reporting a Vulnerability

If you discover a security vulnerability in CryingSense, please:

1. **Do NOT** open a public issue
2. Email the maintainers directly at: [your-email]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

We will respond within 48 hours and work to address the issue promptly.

## Security Scanning

This project uses automated security scanning:
- Dependency vulnerability scanning via GitHub Dependabot
- Code security analysis via CodeQL (when available)
- Regular security audits of dependencies

## Additional Security Considerations

### IoT Deployment
- Secure communication between ESP32 and Raspberry Pi
- Encrypted data transmission to backend servers
- Regular firmware updates for IoT devices
- Network isolation and firewall rules

### Data Privacy
- Infant cry audio contains sensitive data
- Implement proper data encryption at rest and in transit
- Follow GDPR/privacy regulations for data handling
- Secure deletion of processed audio after inference

### Model Security
- Verify model integrity before deployment
- Use model signing and verification
- Protect against adversarial attacks
- Regular model security testing

## References

- [PyTorch Security Advisories](https://github.com/pytorch/pytorch/security/advisories)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [OWASP IoT Security Guidelines](https://owasp.org/www-project-internet-of-things/)
