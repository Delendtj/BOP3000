from dataclasses import dataclass
from tkinter import simpledialog

import cv2
import numpy as np
import tkinter as tk


@dataclass
class LapPanelState:
    panel: np.ndarray | None = None
    key: tuple | None = None

def render_lap_panel(height, width, lap_rows, finish_line_ready, total_laps):
    # Render the current lap summary into the separate lap-counter window.
    panel = np.full((height, width, 3), 24, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (70, 70, 70), 2)
    cv2.putText(
        panel,
        "Lap Counter",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    status_text = "Finish line: ready" if finish_line_ready else "Finish line: press f"
    status_color = (0, 200, 0) if finish_line_ready else (0, 165, 255)
    cv2.putText(
        panel,
        status_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )
    total_laps_text = f"Race laps: {total_laps}"
    cv2.putText(
        panel,
        total_laps_text,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y = 155
    if not lap_rows:
        cv2.putText(
            panel,
            "No active skaters",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
        )
        return panel

    # Keep each row compact so the panel stays readable while updating live.
    for row in lap_rows:
        cv2.rectangle(panel, (16, y - 28), (width - 16, y + 16), (45, 45, 45), -1)
        cv2.putText(
            panel,
            f"ID {row['track_id']}",
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        lap_text = f"{row['lap_count']} / {total_laps}"
        lap_text_size = cv2.getTextSize(lap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        lap_x = width - 28 - lap_text_size[0]
        cv2.putText(
            panel,
            lap_text,
            (lap_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )
        remaining_laps = max(int(total_laps) - int(row["lap_count"]), 0)
        cv2.putText(
            panel,
            f"Remaining: {remaining_laps}",
            (28, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        if row["predicted"]:
            cv2.putText(
                panel,
                "predicted",
                (180, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
        y += 62
        if y > height - 24:
            cv2.putText(
                panel,
                "...",
                (28, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )
            break

    return panel


def update_lap_panel_state(state: LapPanelState, *, tracker, finish_line, height: int, width: int) -> LapPanelState:
    lap_rows = tracker.get_active_lap_counts()
    panel_key = (
        finish_line is not None,
        int(tracker.total_laps),
        tuple(
            (row["track_id"], row["lap_count"], row["predicted"])
            for row in lap_rows
        ),
    )

    if panel_key != state.key:
        state.panel = render_lap_panel(
            height,
            width,
            lap_rows,
            finish_line is not None,
            tracker.total_laps,
        )
        state.key = panel_key

    return state

def prompt_total_laps():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        while True:
            total_laps = simpledialog.askinteger(
                "Race Laps",
                "Enter total laps for this race:",
                minvalue=1,
                parent=root,
            )
            if total_laps is not None:
                return int(total_laps)
    finally:
        root.destroy()
