# API reference excerpt

The relevant compatibility surface for this episode is limited to:

- importing the `requests` package successfully in an environment with urllib3 2.0.7
- preparing a simple GET request without crashing
- constructing the default HTTP adapter without relying on removed urllib3 internals

If these operations succeed, prefer leaving the code unchanged.

