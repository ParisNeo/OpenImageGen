Security and deployment
=======================

OpenImageGen's defaults are suitable for local development, not automatic
exposure to the public internet.

Network exposure
----------------

The default host is ``0.0.0.0``. Use ``127.0.0.1`` for local-only access. Place
production deployments behind a reverse proxy with TLS, request limits, access
control, and appropriate CORS restrictions.

The application enables permissive CORS by default. Restrict that middleware
before exposing a credentialed multi-user deployment to unrelated origins.

Authentication and authorization
--------------------------------

API authentication supports ``X-API-Key`` and ``Authorization: Bearer``.
Configuration can define users, administrator privileges, hourly quotas, and
concurrency limits. Keep configuration files private: stored API keys are
plaintext configuration values.

Model management, plugin reload, and download administration require an
administrator when authentication is enabled. Job status and cancellation check
ownership, but uploaded and generated image URLs are public to anyone who knows
their unguessable identifiers.

The current WebSocket endpoint does not perform the HTTP API-key handshake.
Do not treat it as a private event stream in a hostile deployment.

Credentials in workflows
------------------------

OpenAI-compatible node keys are:

* Masked in the editor.
* Redacted from server-side workflow job details.
* Excluded from exported workflows and reusable node templates.
* Stored in browser-local storage only when the user explicitly opts in.

Browser-local storage is not encrypted. Prefer server environment variables on
shared machines, and never commit keys or Hugging Face tokens.

Uploads and assets
------------------

Image uploads are decoded and re-encoded as PNG and are limited to 10 MiB.
Model assets have extension and size restrictions. Hugging Face downloads are
supported; arbitrary asset URLs are rejected.

These controls do not replace resource isolation. Diffusion models, image
operations, and trusted plugins can consume significant memory, disk, network,
and compute resources.

Operational recommendations
---------------------------

* Use a dedicated service account and environment.
* Limit outbound network access where practical.
* Apply disk quotas for model caches and outputs.
* Keep retention and cleanup enabled.
* Review model repository requirements and licenses.
* Do not enable Python plugins for untrusted users.
* Upgrade dependencies and test model adapters after major Diffusers changes.