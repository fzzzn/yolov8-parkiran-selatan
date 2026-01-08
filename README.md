# YOLOv8 Parking Monitor

Automated motorcycle parking monitoring system using YOLOv8 detection with Telegram notifications.

## Features

- Real-time RTSP camera feed processing
- YOLOv8x model for accurate motorcycle detection
- Two parking areas monitoring (Orange and Red zones)
- Presence detection (DETECTED/EMPTY status)
- Periodic capture and analysis (every 60 seconds)
- Telegram notifications with annotated images
- Docker support for easy deployment

## Requirements

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- RTSP camera feed
- Telegram Bot token and Chat ID

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your credentials:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   RTSP_URL=rtsp://your_camera_ip/live/0/MAIN
   ```

3. Define parking areas in `areas.py` (use `define_areas.py` tool for interactive setup)

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

## Docker Deployment

### Using Docker Compose (Recommended)

1. Build and start the container:
   ```bash
   docker-compose up -d
   ```

2. View logs:
   ```bash
   docker-compose logs -f
   ```

3. Stop the container:
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. Build the image:
   ```bash
   docker build -t yolov8-parking .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name yolov8-parking-monitor \
     --env-file .env \
     --restart unless-stopped \
     yolov8-parking
   ```

## Detection Parameters

- **Model**: YOLOv8x (extra-large for maximum accuracy)
- **Confidence**: 0.01 (ultra-low for maximum detection)
- **IOU Threshold**: 0.10 (high overlap tolerance)
- **Image Size**: 1536px
- **Max Detections**: 1000

## Project Structure

```
.
├── main.py              # Main application logic
├── areas.py            # Parking area polygon definitions
├── telegram_bot.py     # Telegram integration
├── define_areas.py     # Interactive area definition tool
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker Compose configuration
├── .env              # Environment variables (not in git)
└── .env.example      # Environment template
```

## License

MIT
