"""The SBIRT clinical protocol: instruments, wording, and the engine that runs them.

Deliberately empty. config imports this package for the study wording, so
anything re-exported here that in turn imports config would close a cycle —
every consumer imports the submodule it needs directly.
"""
