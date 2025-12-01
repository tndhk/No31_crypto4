# Live Trading Setup Guide

## 1. Prerequisites

### Binance Account
- Create a Binance account if you haven't already.
- Enable Two-Factor Authentication (2FA).
- **API Management**:
    - Create a new API Key.
    - Enable "Enable Spot & Margin Trading".
    - **IMPORTANT**: Restrict IP access to your server's IP address for security.
    - Copy the `API Key` and `Secret Key`.

## 2. Environment Configuration

Create a `.env` file in the project root directory (`/Users/takahiko_tsunoda/work/dev/crypt4/.env`) and add your keys:

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_key_here
```

> **Note**: The system loads these variables automatically if you use a tool like `python-dotenv`, or you can export them in your shell.
> Currently, the system expects them to be set in the environment.

**Recommended: Export in Shell (for testing)**
```bash
export BINANCE_API_KEY=your_api_key_here
export BINANCE_SECRET=your_secret_key_here
```

## 3. Execution

### Dry Run (Test Mode)
Verify everything works without placing real orders:

```bash
venv/bin/python -m src.cli --mode live --dry-run --config config/settings.yaml
```

### Live Trading (Real Money)
**WARNING**: This will execute real trades with your funds.

```bash
venv/bin/python -m src.cli --mode live --config config/settings.yaml
```

## 4. Operation

- The bot runs a scheduler that checks for signals every hour.
- Logs are printed to the console and saved in `logs/` (if configured).
- To stop the bot, press `Ctrl+C`.

## 5. Safety Checks

- **Leverage**: Ensure `execution.leverage` in `config/settings.yaml` is set appropriately.
- **Capital**: The bot currently uses a fixed allocation logic (50% of available USDT balance per trade in the example logic). Adjust `src/scheduler.py` if needed.
