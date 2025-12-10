import asyncio
import os
import re
import random
import pandas as pd
from playwright.async_api import async_playwright

# Expanded valid positions with refinements
VALID_POSITIONS = [
    "QB",
    "HB", "FB",
    "WR", "TE",
    "LT", "LG", "C", "RG", "RT",
    "DT",
    "SAM", "MIKE", "WILL",
    "CB",
    "SS", "FS",
    "K", "P",
    "REDG", "LEDG"
]

# Dealbreaker mapping
DEALBREAKER_MAP = {
    "None": "15",
    "Brand Exposure": "2",
    "Championship Contender": "4",
    "Coach Prestige": "5",
    "Conference Prestige": "7",
    "Playing Style": "8",
    "Playing Time": "9",
    "Pro Potential": "10",
    "Proximity To Home": "12"
}

async def main():
    roster_file = input("Enter the path to your roster Excel file (e.g., Team1.xlsx): ").strip()
    if not os.path.isfile(roster_file):
        print(f"Error: {roster_file} not found.")
        return

    roster_df = pd.read_excel(roster_file)
    print(f"Loaded roster with {len(roster_df)} players.")

    # Normalize spreadsheet positions
    roster_df["Position"] = roster_df["Position"].str.upper().replace({
        "RB": "HB",
        "LOLB": "SAM",
        "MLB": "MIKE",
        "ROLB": "WILL",
        "RE": "REDG",
        "LE": "LEDG"
    })

    # Track how many players have been used per position
    position_counters = {pos: 0 for pos in VALID_POSITIONS}

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        page = await browser.new_page()

        # --- Login flow (manual) ---
        await page.goto("https://www.ea.com/games/ea-sports-college-football/team-builder/teams/my-teams")
        print("⚠️ Please log in manually in the browser window.")
        print("Enter your email and password directly on the EA login page.")
        input("Press ENTER here once you have successfully logged in...")

        # After manual login, continue navigation
        await page.click("div[routerlink='my-teams']")
        await asyncio.sleep(5)

        # --- Wait for roster page ---
        await page.wait_for_selector("div.player-ticket--info", timeout=0)

        # --- Scroll to load all players ---
        scroll_container = page.locator("div.team-create-sidebar.scroll-container").first
        for _ in range(10):
            await scroll_container.evaluate("el => el.scrollBy(0, 500)")
            await asyncio.sleep(1)

        # --- Pass 2: Fill bio form fields using explicit selectors ---
        player_cards = page.locator("button.player-ticket")
        count = await player_cards.count()
        print(f"Found {count} players on roster page.")

        for i in range(count):
            player_cards = page.locator("button.player-ticket")
            card = player_cards.nth(i)
            await card.click()
            await asyncio.sleep(2)

            info_block = card.locator("div.player-ticket--info")
            await info_block.wait_for(timeout=0)

            spans = info_block.locator("span.fs-sm.lh-sm.fw-400.c-text-white")
            count_spans = await spans.count()

            position = None
            for j in range(count_spans):
                text = (await spans.nth(j).text_content()).strip().upper()
                match_pos = re.search(r"\b(QB|HB|FB|WR|TE|LT|LG|C|RG|RT|DT|SAM|MIKE|WILL|CB|SS|FS|K|P|REDG|LEDG)\b", text)
                if match_pos:
                    position = match_pos.group(1)
                    break

            if not position:
                print(f"No position found for card {i}")
                continue

            matches = roster_df[roster_df["Position"].str.upper() == position]
            idx = position_counters[position]

            if idx >= len(matches):
                print(f"No more spreadsheet players left for position {position}")
                continue

            match = matches.iloc[idx]
            position_counters[position] += 1

            # First Name
            await page.locator("form.player-bio-grid input#textInput").nth(0).fill(str(match["FirstName"]))

            # Last Name
            await page.locator("form.player-bio-grid input#textInput").nth(1).fill(str(match["LastName"]))

            # --- Check for profanity flag ---
            try:
                await page.wait_for_selector("div.error-message", timeout=3000)
                print(f"⚠️ Potential profanity flag detected for {match['FirstName']} {match['LastName']}.")
                print("Please fix the name manually in the browser. Press ENTER here when ready to continue...")
                input()
            except:
                pass  # No error message appeared

            # Jersey Number (dropdown)
            await page.locator("form.player-bio-grid select#weight").nth(0).select_option(str(match["Jersey"]))

            # Previous Redshirt (dropdown)
            if "Redshirt" in match.index:
                if str(match["Redshirt"]).lower() == "yes":
                    await page.locator("form.player-bio-grid select#weight").nth(1).select_option("3")
                else:
                    await page.locator("form.player-bio-grid select#weight").nth(1).select_option("0")

            # Handedness (dropdown)
            await page.locator("form.player-bio-grid select#weight").nth(2).select_option(
                "1" if str(match["Handedness"]).lower() == "left" else "0"
            )

            # Year (dropdown)
            year_map = {"FRESHMAN": "0", "SOPHOMORE": "1", "JUNIOR": "2", "SENIOR": "3"}
            await page.locator("form.player-bio-grid select#weight").nth(3).select_option(
                year_map.get(str(match["Year"]).upper(), "0")
            )

            # Height slider
            height_val = max(65, min(84, int(round(float(match["Height_in"])))))
            await page.locator("input#heightSlider").fill(str(height_val))

            # Weight slider
            weight_val = max(160, min(400, int(round(float(match["Weight_lb"])))))
            await page.locator("input#weightSlider").fill(str(weight_val))

            # HS star rating
            stars = page.locator("div.grade-wrapper button")
            if "hs star rating" in match.index:
                rating_raw = str(match["hs star rating"]).strip().lower()
                word_to_num = {"one":1,"two":2,"three":3,"four":4,"five":5}
                if rating_raw.isdigit():
                    rating = int(rating_raw)
                elif rating_raw in word_to_num:
                    rating = word_to_num[rating_raw]
                else:
                    rating = 0
                if rating > 0:
                    await stars.nth(rating - 1).click()

            # Dealbreaker
            if "dealbreaker" in match.index:
                db_raw = str(match["dealbreaker"]).strip()
                db_value = DEALBREAKER_MAP.get(db_raw, "15")
                await page.locator("form.player-bio-grid select#weight").nth(4).select_option(db_value)

            # Random skin tone
            color_buttons = page.locator("div.filterGroup--colors button.color-btn")
            count_colors = await color_buttons.count()
            if count_colors > 0:
                choice = random.randint(0, count_colors - 1)
                await color_buttons.nth(choice).click()

            # Random head selector
            head_buttons = page.locator("div.filterGroup--design--xl button.design-btn")
            count_heads = await head_buttons.count()
            if count_heads > 0:
                choice = random.randint(0, count_heads - 1)
                await head_buttons.nth(choice).click()

            await asyncio.sleep(2)

        # Save once at the end
        await page.click("button:has-text('Save')")
        await asyncio.sleep(5)
        print("Roster assignment and bio filling complete. Final roster saved.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
