import os
import unittest


class NoLiveKeysInTests(unittest.TestCase):
    """Importing the app in a test run must not load .env.local (it booked real calendar events)."""

    def test_env_local_is_not_loaded(self):
        before = dict(os.environ)
        import server

        server._load_env_local()
        for key in ("GCAL_SCRIPT_URL", "TELEGRAM_BOT_TOKEN", "TWILIO_AUTH_TOKEN"):
            self.assertEqual(os.environ.get(key), before.get(key), key)
            self.assertFalse(os.environ.get(key, "").startswith("https://script.google.com"), key)


if __name__ == "__main__":
    unittest.main()
