# Logging

RepTate uses a centralized logging configuration with rotating files and console
output. Logs are written to a platform-specific application data directory.

## Default locations

- Linux: `~/.local/share/RepTate/`
- macOS/Windows: Qt `AppDataLocation` (when available)

Files created:
- `RepTate.log` (info/debug)
- `RepTate.error.log` (errors and above)

## Environment overrides

- `REPTATE_LOG_DIR`: override log directory
- `REPTATE_LOG_LEVEL`: main logger level (e.g., INFO, DEBUG)
- `REPTATE_LOG_CONSOLE_LEVEL`: console level (default WARNING)
- `REPTATE_LOG_FILE_LEVEL`: file level (default main level)
- `REPTATE_LOG_ERROR_LEVEL`: error file level (default ERROR)
- `REPTATE_LOG_MAX_BYTES`: rotation size (default 2000000)
- `REPTATE_LOG_BACKUP_COUNT`: number of backups (default 5)

## Notes

- GUI log output is also mirrored to the in-app logger window.
- Unhandled exceptions are captured and written to the log files.
