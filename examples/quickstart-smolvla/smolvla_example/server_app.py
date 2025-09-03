"""Flower server app for SmolVLA federated learning."""

import flwr as fl
from flwr.server import ServerApp, ServerConfig


# Create server app
app = ServerApp(
    config=ServerConfig(
        num_rounds=100,  # For good results, total training rounds should be over 100,000
    )
)


def main() -> None:
    """Run the SmolVLA federated learning server."""
    # Start server
    app.run()


if __name__ == "__main__":
    main()