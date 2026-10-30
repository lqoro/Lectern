"""The Mac app's update plumbing: what Info.plist promises Sparkle, the feed the
release script writes, and that the Settings section only shows where a native
updater exists. Run: python -m unittest discover -s tests"""
import base64
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging" / "macos"))
import build_app  # noqa: E402
import make_appcast  # noqa: E402

SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"


class InfoPlist(unittest.TestCase):
    def setUp(self):
        self.plist = build_app.info_plist()

    def test_updates_are_signed_and_fetched_over_https(self):
        self.assertTrue(build_app.FEED_URL.startswith("https://"))
        key = base64.b64decode(self.plist["SUPublicEDKey"])
        self.assertEqual(len(key), 32, "an Ed25519 public key is 32 bytes")

    def test_nothing_installs_without_the_user_and_no_profile_is_sent(self):
        self.assertIs(self.plist["SUAllowsAutomaticUpdates"], False)
        self.assertIs(self.plist["SUEnableSystemProfiling"], False)

    def test_the_two_version_fields_agree_so_an_install_is_not_offered_again(self):
        self.assertEqual(self.plist["CFBundleShortVersionString"], self.plist["CFBundleVersion"])


class Feed(unittest.TestCase):
    def feed(self, notes="<p>Fixes.</p>", url="https://example.com/a&b/Lectern-1.2.3.dmg"):
        return ET.fromstring(make_appcast.feed_xml("1.2.3", notes, url, "c2ln+/==", 4242))

    def test_the_item_carries_what_sparkle_checks(self):
        item = self.feed().find("channel/item")
        self.assertEqual(item.findtext(SPARKLE + "version"), "1.2.3")
        self.assertEqual(item.findtext(SPARKLE + "minimumSystemVersion"), build_app.MIN_MACOS)
        enc = item.find("enclosure")
        self.assertEqual(enc.get("url"), "https://example.com/a&b/Lectern-1.2.3.dmg")
        self.assertEqual(enc.get(SPARKLE + "edSignature"), "c2ln+/==")
        self.assertEqual(enc.get("length"), "4242")

    def test_release_notes_are_html_inside_the_item(self):
        item = self.feed(notes="<ul><li>A &amp; B</li></ul>").find("channel/item")
        self.assertEqual(item.findtext("description"), "<ul><li>A &amp; B</li></ul>")

    def test_notes_that_would_end_the_cdata_block_are_refused(self):
        with self.assertRaises(SystemExit):
            make_appcast.feed_xml("1.2.3", "x ]]> y", "https://e.com/x.dmg", "s", 1)


class SettingsPage(unittest.TestCase):
    def test_updates_are_hidden_until_the_app_answers(self):
        html = (ROOT / "web" / "index.html").read_text()
        self.assertIn('id="updatesSec"', html)
        self.assertRegex(html, r'<section class="set-sec hidden" id="updatesSec">')

    def test_the_page_only_calls_a_bridge_that_exists(self):
        js = (ROOT / "web" / "app.js").read_text()
        self.assertIn("window.webkit.messageHandlers.lectern", js)
        self.assertIn("if (bridge)", js)


if __name__ == "__main__":
    unittest.main()
