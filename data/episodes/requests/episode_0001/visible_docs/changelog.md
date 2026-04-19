# urllib3 2.0 compatibility note

urllib3 2.x keeps the high-level connection pooling APIs but removes a number of
deprecated aliases and tightens several internal behaviors. Client libraries that
already depend on the public request and response interfaces usually do not need
large code changes.

For this benchmark episode, treat the main risk as verification: confirm that the
host application still imports cleanly and that prepared request behavior remains
stable under urllib3 2.0.7.

