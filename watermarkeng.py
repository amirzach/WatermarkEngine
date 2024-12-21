import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image, ImageDraw, ImageFont

def dct2(block):
    """Perform 2D DCT."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Perform 2D IDCT."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def create_text_watermark(text, image_shape, font_path="arial.ttf", font_size=20):
    """Create a repeating text watermark to cover the entire image."""
    h, w = image_shape[:2]
    watermark_image = Image.new('L', (w // 8, h // 8), color=0)
    draw = ImageDraw.Draw(watermark_image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    for i in range(0, watermark_image.size[1], text_height + 10):
        for j in range(0, watermark_image.size[0], text_width + 10):
            draw.text((j, i), text, fill=255, font=font)

    return np.array(watermark_image, dtype=np.uint8)

def embed_watermark_dct_dwt(host_image, watermark, alpha=10):
    """Embed a watermark into a color host image using DCT and DWT."""
    # Ensure the host image is in color
    if len(host_image.shape) != 3 or host_image.shape[2] != 3:
        raise ValueError("Host image must be a color image (3 channels).")

    h, w, _ = host_image.shape
    watermarked_image = host_image.copy().astype(float)

    # Apply Gaussian blur to reduce pixelation
    watermark_smoothed = cv2.GaussianBlur(watermark, (5, 5), 0)

    watermark_normalized = watermark_smoothed / 255.0  # Normalize to range [0, 1]

    # Process each color channel
    for channel in range(3):  # Loop through R, G, B channels
        # Apply DWT to the channel
        coeffs = pywt.dwt2(host_image[:, :, channel], 'haar')
        LL, (LH, HL, HH) = coeffs

        # Process LL (approximation) subband with DCT
        h_LL, w_LL = LL.shape
        for i in range(0, h_LL, 8):
            for j in range(0, w_LL, 8):
                block = LL[i:i+8, j:j+8]
                if block.shape[0] < 8 or block.shape[1] < 8:
                    continue  # Skip blocks smaller than 8x8

                dct_block = dct2(block)

                # Embed watermark in the (4, 4) coefficient
                watermark_bit = watermark_normalized[i // 8, j // 8]
                dct_block[4, 4] += alpha * watermark_bit

                # Reconstruct the block using IDCT
                LL[i:i+8, j:j+8] = idct2(dct_block)

        # Reconstruct the channel with inverse DWT
        watermarked_image[:, :, channel] = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark_dct_dwt(host_image, watermarked_image, alpha=10):
    """Extract the embedded watermark from a color watermarked image using DCT and DWT."""
    h, w, _ = host_image.shape
    extracted_watermark = np.zeros((h // 8, w // 8), dtype=float)

    # Process the red channel for watermark extraction
    coeffs_host = pywt.dwt2(host_image[:, :, 0], 'haar')
    coeffs_watermarked = pywt.dwt2(watermarked_image[:, :, 0], 'haar')

    LL_host, _ = coeffs_host
    LL_watermarked, _ = coeffs_watermarked

    h_LL, w_LL = LL_host.shape
    for i in range(0, h_LL, 8):
        for j in range(0, w_LL, 8):
            orig_block = LL_host[i:i+8, j:j+8]
            watermarked_block = LL_watermarked[i:i+8, j:j+8]

            if orig_block.shape[0] < 8 or orig_block.shape[1] < 8:
                continue  # Skip blocks smaller than 8x8

            orig_dct_block = dct2(orig_block)
            watermarked_dct_block = dct2(watermarked_block)

            # Extract the watermark bit
            difference = (watermarked_dct_block[4, 4] - orig_dct_block[4, 4])
            extracted_watermark[i // 8, j // 8] = difference / alpha

    # Scale the extracted watermark back to 0-255
    extracted_watermark = np.clip(extracted_watermark, 0, 1) * 255
    return extracted_watermark.astype(np.uint8)

# Main Execution
if __name__ == "__main__":
    host_image_path = r"C:\Users\User\tiger.jpg"
    text_watermark = "NAZIR"

    # Load the host image
    host_image = cv2.imread(host_image_path)  # Load as color image
    if host_image is None:
        raise ValueError("Host image not found or invalid format.")

    # Generate the text watermark
    watermark_image = create_text_watermark(text_watermark, host_image.shape, font_size=30)

    # Embed the watermark
    watermarked_image = embed_watermark_dct_dwt(host_image, watermark_image, alpha=20000)
    output_path = r"C:\Users\User\watermarked_image.png"
    cv2.imwrite(output_path, watermarked_image)
    print(f"Watermark embedded and saved as '{output_path}'.")

    # Extract the watermark
    extracted_watermark = extract_watermark_dct_dwt(host_image, watermarked_image, alpha=20000)
    extracted_path = r"C:\Users\User\extracted_watermark.png"
    cv2.imwrite(extracted_path, extracted_watermark)
    print(f"Watermark extracted and saved as '{extracted_path}'.")
