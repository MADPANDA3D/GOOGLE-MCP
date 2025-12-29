import argparse
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow


def parse_scopes(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.replace(",", " ")
    return [scope.strip() for scope in cleaned.split() if scope.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local OAuth flow and write a token.json for Google APIs."
    )
    parser.add_argument(
        "--credentials",
        default="fastmcp/.google/credentials.json",
        help="Path to OAuth client credentials.json",
    )
    parser.add_argument(
        "--token",
        default="fastmcp/.google/token.json",
        help="Path to write the user token.json",
    )
    parser.add_argument(
        "--scopes",
        default=os.getenv("GOOGLE_SCOPES", ""),
        help="Space or comma separated OAuth scopes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local server port for the OAuth callback (0 = auto)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically",
    )
    args = parser.parse_args()

    scopes = parse_scopes(args.scopes)
    if not scopes:
        print("No scopes provided. Use --scopes or set GOOGLE_SCOPES in your shell.")
        return 1

    if not os.path.exists(args.credentials):
        print(f"Missing credentials file: {args.credentials}")
        return 1

    token_dir = os.path.dirname(args.token)
    if token_dir:
        os.makedirs(token_dir, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(args.credentials, scopes=scopes)
    creds = flow.run_local_server(port=args.port, open_browser=not args.no_browser)

    with open(args.token, "w", encoding="utf-8") as handle:
        handle.write(creds.to_json())

    print(f"Wrote token to {args.token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
