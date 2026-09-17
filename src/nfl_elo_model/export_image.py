"""Render the weekly predictions table as a shareable PNG."""

from io import BytesIO

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#141f2e"
ROW_DARK = "#1a293a"
ROW_LIGHT = "#1e2e40"
HEADER = "#294352"
TEXT = "#f3f5fa"
MUTED = "#c5cbd5"
ACCENT = "#96ecd5"
MIN_WIDTHS = {
    "Date": 160,
    "Matchup": 580,
    "Status": 162,
    "Projected winner": 288,
    "Win probability": 214,
    "Home spread": 196,
    "Projected total": 208,
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled default font is available in the slim deployment image.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def weekly_predictions_png(table: pd.DataFrame, season: int, week: int) -> bytes:
    """Return a PNG containing exactly the rows and columns shown in the app."""
    body_font = _font(22)
    title_font = _font(40)
    label_font = _font(20)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    columns = list(table.columns)
    widths = [
        max(
            MIN_WIDTHS.get(column, 160),
            max([measure.textlength(str(column), font=body_font)] +
                [measure.textlength(str(value), font=body_font) for value in table[column]]) + 48,
        )
        for column in columns
    ]
    left, top, row_height = 40, 174, 62
    width = max(1728, left * 2 + int(sum(widths)))
    height = top + row_height * (len(table) + 1) + 76
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((left, 28), "NFL OUTCOME PREDICTOR", font=label_font, fill=ACCENT)
    draw.text((left, 69), f"Weekly predictions | {season} | Week {week}", font=title_font, fill=TEXT)
    draw.text((left, 127), f"{len(table)} games | Based on the saved schedule", font=label_font, fill=MUTED)

    x_end = left + int(sum(widths))
    draw.rectangle((left, top, x_end, top + row_height - 1), fill=HEADER)
    for row_number, (_, row) in enumerate(table.iterrows()):
        y = top + row_height * (row_number + 1)
        draw.rectangle((left, y, x_end, y + row_height - 1), fill=ROW_LIGHT if row_number % 2 == 0 else ROW_DARK)
        x = left
        for column, cell_width in zip(columns, widths):
            color = ACCENT if column in ("Projected winner", "Win probability") else TEXT
            draw.text((x + 24, y + row_height / 2), str(row[column]), font=body_font, fill=color, anchor="lm")
            x += cell_width

    x = left
    for column, cell_width in zip(columns, widths):
        draw.text((x + 24, top + row_height / 2), str(column), font=body_font, fill=TEXT, anchor="lm")
        x += cell_width

    draw.text(
        (left, height - 43),
        "Home spread: negative favors home; positive favors away.",
        font=label_font,
        fill=MUTED,
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
