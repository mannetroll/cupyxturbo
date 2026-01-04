import pathlib, re

inp = pathlib.Path("gpu-profile.txt")
data = inp.read_bytes()

# Normalize line endings at the BYTES level: CRLF/CR -> LF
data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

# --- Strip ANSI escape sequences (bytes) ---

# OSC: ESC ] ... BEL  OR  ESC ] ... ESC \
ansi_osc = re.compile(rb"\x1b\][^\x07]*(?:\x07|\x1b\\)")

# DCS: ESC P ... ESC \
ansi_dcs = re.compile(rb"\x1bP.*?\x1b\\", re.DOTALL)

# CSI: ESC [ ... final_byte   OR single-byte CSI 0x9b ... final_byte
ansi_csi = re.compile(rb"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")

# 2-byte ESC sequences (often used for simple cursor/charset controls)
ansi_esc_2byte = re.compile(rb"\x1b[@-Z\\-_]")

data = ansi_osc.sub(b"", data)
data = ansi_dcs.sub(b"", data)
data = ansi_csi.sub(b"", data)
data = ansi_esc_2byte.sub(b"", data)

# Decode: try UTF-8 first (prevents â?? mojibake), then fall back
for enc in ("utf-8", "utf-8-sig", "cp1252"):
    try:
        text = data.decode(enc)
        break
    except UnicodeDecodeError:
        pass
else:
    text = data.decode("utf-8", errors="replace")

# Normalize any remaining odd line separators at the TEXT level
text = text.replace("\u0085", "\n")   # NEL
text = text.replace("\r\n", "\n").replace("\r", "\n")

# Some tools render ESC as literal "←[" or "^["; strip full sequences in that form too
arrow_csi = re.compile(r"(?:\u2190|\^)\[[0-?]*[ -/]*[@-~]")
text = arrow_csi.sub("", text)

out = pathlib.Path("gpu-profile.clean.utf8.txt")
out.write_text(text, encoding="utf-8", newline="\n")
print("Wrote", out)
