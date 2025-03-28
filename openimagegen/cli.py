import argparse
import uvicorn
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="OpenImageGen API Server")
    parser.add_argument("--host", default=None, help="Host to bind the server to (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind the server to (overrides config)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development (requires 'watchfiles')")
    parser.add_argument("--config", type=str, help="Path to the config.toml file (overrides env var and default paths)")
    args = parser.parse_args()

    # --- Configuration Loading Logic (Simplified for CLI entry point) ---
    # Priority: CLI arg > Env Var > Standard Paths > Default
    config_path_to_set = None
    if args.config:
        config_path_to_set = Path(args.config).resolve()
        if not config_path_to_set.exists():
            print(f"Warning: Config file specified via --config not found: {config_path_to_set}")
            # Proceed, main.py will handle default creation or raise error if loading fails later
    elif os.getenv("OPENIMAGEGEN_CONFIG"):
        config_path_to_set = Path(os.getenv("OPENIMAGEGEN_CONFIG")).resolve()
        if not config_path_to_set.exists():
            print(f"Warning: Config file specified via OPENIMAGEGEN_CONFIG not found: {config_path_to_set}")
            # Proceed as above

    # Set environment variable for main.py to pick up, regardless of source
    # This centralizes config path resolution before the app starts
    if config_path_to_set:
        os.environ["OPENIMAGEGEN_CONFIG_OVERRIDE"] = str(config_path_to_set)
    # -------------------------------------------------------------------

    # Load host/port from config *if not* specified via CLI
    # This requires loading the config here, slightly redundant but needed for Uvicorn args
    host = args.host
    port = args.port
    if host is None or port is None:
        try:
            # Use the same loading logic as in main.py to find the config
            from .main import load_config_for_cli # Import specific function
            config_data = load_config_for_cli()
            if host is None:
                host = config_data.get("settings", {}).get("host", "0.0.0.0")
            if port is None:
                port = config_data.get("settings", {}).get("port", 8089)
        except Exception as e:
            print(f"Could not load config to determine host/port, using defaults: {e}")
            if host is None: host = "0.0.0.0"
            if port is None: port = 8089


    # Run the Uvicorn server programmatically
    print(f"Starting OpenImageGen server on {host}:{port}")
    uvicorn.run(
        app="openimagegen.main:app",  # The FastAPI app
        host=host,
        port=port,
        reload=args.reload,
        # reload_dirs=["openimagegen"] if args.reload else None # Optional: specify dirs for reload
    )

if __name__ == "__main__":
    main()
