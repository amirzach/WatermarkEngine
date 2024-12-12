import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct

def dct2(block):
    """Perform 2D DCT."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Perform 2D IDCT."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_watermark_dct_dwt(host_image, watermark, alpha=10):
    """Embed a binary watermark into a color host image using DCT and DWT."""
    # Ensure the host image is in color
    if len(host_image.shape) != 3 or host_image.shape[2] != 3:
        raise ValueError("Host image must be a color image (3 channels).")

    h, w, _ = host_image.shape
    watermarked_image = host_image.copy().astype(float)

    # Resize watermark to match the dimensions of host image channels
    watermark_resized = cv2.resize(watermark, (w // 8, h // 8))
    watermark_binary = (watermark_resized > 128).astype(int)  # Convert to binary

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
                dct_block = dct2(block)

                # Embed watermark in the (4, 4) coefficient
                watermark_bit = watermark_binary[i // 8, j // 8]
                dct_block[4, 4] += alpha * watermark_bit

                # Reconstruct the block using IDCT
                LL[i:i+8, j:j+8] = idct2(dct_block)

        # Reconstruct the channel with inverse DWT
        watermarked_image[:, :, channel] = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark_dct_dwt(host_image, watermarked_image, alpha=10):
    """Extract the embedded watermark from a color watermarked image using DCT and DWT."""
    h, w, _ = host_image.shape
    extracted_watermark = np.zeros((h // 8, w // 8), dtype=int)

    # Process the red channel for watermark extraction
    coeffs_host = pywt.dwt2(host_image[:, :, 0], 'haar')
    coeffs_watermarked = pywt.dwt2(watermarked_image[:, :, 0], 'haar')

    LL_host, _ = coeffs_host
    LL_watermarked, _ = coeffs_watermarked

    h_LL, w_LL = LL_host.shape
    for i in range(0, h_LL, 8):
        for j in range(0, w_LL, 8):
            orig_block = dct2(LL_host[i:i+8, j:j+8])
            watermarked_block = dct2(LL_watermarked[i:i+8, j:j+8])

            # Extract the watermark bit
            difference = (watermarked_block[4, 4] - orig_block[4, 4])
            extracted_watermark[i // 8, j // 8] = round(difference / alpha)

    # Scale the extracted watermark back to 0-255
    return (extracted_watermark * 255).astype(np.uint8)

# Main Execution
if __name__ == "__main__":
    host_image_path = r"C:\Users\User\host_image.png"
    watermark_image_path = r"C:\Users\User\watermark_image.png"

    # Load the host and watermark images
    host_image = cv2.imread(host_image_path)  # Load as color image
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Embed the watermark
    watermarked_image = embed_watermark_dct_dwt(host_image, watermark_image, alpha=200)
    cv2.imwrite(r"C:\Users\User\watermarked_image.png", watermarked_image)
    print("Watermark embedded and saved as 'watermarked_image.png' in User.")

    # Extract the watermark
    extracted_watermark = extract_watermark_dct_dwt(host_image, watermarked_image, alpha=200)
    cv2.imwrite(r"C:\Users\User\extracted_watermark.jpg", extracted_watermark)
    print("Watermark extracted and saved as 'extracted_watermark.jpg' in User.")
