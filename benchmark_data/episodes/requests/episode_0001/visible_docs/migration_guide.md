# Migration guide excerpt

- Prefer public urllib3 interfaces over deprecated internal helpers.
- Avoid pinning to legacy urllib3 1.26-only implementation details.
- If the application already allows `urllib3<3`, start by validating behavior
  before changing code.
- Keep code churn low when the upgraded dependency is already compatible.

