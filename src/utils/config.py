import configparser
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_prop_betting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('mlb_prop_betting')

def load_config():
    """Load configuration from config.ini file or environment variables"""
    # First try to find a config file
    config = configparser.ConfigParser()
    if os.path.exists('config.ini'):
        config.read('config.ini')
        try:
            return {
                'db_host': config.get('Database', 'host', fallback='localhost'),
                'db_name': config.get('Database', 'database', fallback='mlb_prop_betting'),
                'db_user': config.get('Database', 'user', fallback='postgres'),
                'db_password': config.get('Database', 'password', fallback='ophadke'),
                'api_base_url': config.get('API', 'base_url', fallback='https://statsapi.mlb.com/api/v1'),
                'sportsbook_api_key': config.get('API', 'sportsbook_api_key', fallback=''),
                'sportsbook_api_url': config.get('API', 'sportsbook_api_url', fallback=''),
            }
        except Exception as e:
            logger.warning(f"Error reading config file: {e}")
    
    # Fall back to environment variables or default values
    return {
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_name': os.getenv('DB_NAME', 'mlb_prop_betting'),
        'db_user': os.getenv('DB_USER', 'postgres'),
        'db_password': os.getenv('DB_PASSWORD', 'ophadke'),  # Don't hardcode in production
        'api_base_url': os.getenv('API_BASE_URL', 'https://statsapi.mlb.com/api/v1'),
        'sportsbook_api_key': os.getenv('SPORTSBOOK_API_KEY', ''),
        'sportsbook_api_url': os.getenv('SPORTSBOOK_API_URL', ''),
    }
