"""
@file        cli.py
@author      ParisNeo
@description Command-line interface for OpenImageGen.
             Handles argument parsing, configuration overrides, and starting the API server.
"""

import argparse
import os
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="OpenImageGen API Server")
    parser.add_argument("--config", type=str, default=None, help="Path to the config.toml file")
    subparsers = parser.add_subparsers(dest="command", help="OpenImageGen commands")

    serve_parser = subparsers.add_parser("serve", help="Start the API server (default action)")
    serve_parser.add_argument("--host", default=None, help="Host to bind the server to (overrides config)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind the server to (overrides config)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    download_parser = subparsers.add_parser("download", help="Pre-download model weights without starting the server")
    download_parser.add_argument("--models", nargs="*", help="Specific model keys to download. If omitted, downloads all configured models.")

    args = parser.parse_args()

    config_path_to_set = None
    if args.config:
        config_path_to_set = Path(args.config).resolve()
        if not config_path_to_set.exists():
            print(f"Warning: Config file specified via --config not found: {config_path_to_set}")
    elif os.getenv("OPENIMAGEGEN_CONFIG"):
        config_path_to_set = Path(os.getenv("OPENIMAGEGEN_CONFIG")).resolve()
        if not config_path_to_set.exists():
            print(f"Warning: Config file specified via OPENIMAGEGEN_CONFIG not found: {config_path_to_set}")

    if config_path_to_set:
        os.environ["OPENIMAGEGEN_CONFIG_OVERRIDE"] = str(config_path_to_set)

    if args.command == "download":
        from .service import ImageGenService
        print("Initializing service for model download...")
        svc = ImageGenService()
        target_models = args.models if args.models else None
        if target_models:
            print(f"Starting download for specific models: {', '.join(target_models)}")
        else:
            print("Starting download for all configured models...")
        
        results = svc.preload_models(model_keys=target_models)
        
        print("\nDownload Summary:")
        for key, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {key}: {status}")
        return

    host = args.host if hasattr(args, 'host') else None
    port = args.port if hasattr(args, 'port') else None
    if host is None or port is None:
        try:
            from .config import load_config
            config_data = load_config(suppress_logs=True)
            if host is None:
                host = config_data.get("settings", {}).get("host", "0.0.0.0")
            if port is None:
                port = config_data.get("settings", {}).get("port", 8089)
        except Exception as e:
            print(f"Could not load config to determine host/port, using defaults: {e}")
            if host is None:
                host = "0.0.0.0"
            if port is None:
                port = 8089

    print(f"Starting OpenImageGen server on {host}:{port}")
    uvicorn.run(
        app="openimagegen.app:app",
        host=host,
        port=port,
        reload=args.reload if hasattr(args, 'reload') else False,
    )


if __name__ == "__main__":
    main()