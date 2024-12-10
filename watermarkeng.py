import cv2
import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    """Perform 2D DCT."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Perform 2D IDCT."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_watermark(host_image, watermark, alpha=10):
    """Embed a binary watermark into the host image using DCT."""
    # Convert host image to grayscale if it is colored
    if len(host_image.shape) == 3:
        host_image = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
    
    h, w = host_image.shape
    watermarked_image = host_image.copy().astype(float)
    
    # Resize watermark to match the dimensions of host image
    watermark_resized = cv2.resize(watermark, (w // 8, h // 8))
    watermark_binary = (watermark_resized > 128).astype(int)  # Convert to binary

    # Process the image in 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = host_image[i:i+8, j:j+8]
            dct_block = dct2(block)
            
            # Embed watermark in the (4, 4) coefficient
            watermark_bit = watermark_binary[i // 8, j // 8]
            dct_block[4, 4] += alpha * watermark_bit
            
            # Reconstruct the block using IDCT
            watermarked_image[i:i+8, j:j+8] = idct2(dct_block)
    
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark(host_image, watermarked_image, alpha=10):
    """Extract the embedded watermark from the watermarked image."""
    h, w = host_image.shape
    extracted_watermark = np.zeros((h // 8, w // 8), dtype=int)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            orig_block = dct2(host_image[i:i+8, j:j+8])
            watermarked_block = dct2(watermarked_image[i:i+8, j:j+8])
            
            # Extract the watermark bit
            difference = (watermarked_block[4, 4] - orig_block[4, 4])
            extracted_watermark[i // 8, j // 8] = round(difference / alpha)
    
    # Scale the extracted watermark back to 0-255
    return (extracted_watermark * 255).astype(np.uint8)

# Main Execution
if __name__ == "__main__":
    # Replace <YourUsername> with your actual username
    host_image_path = r"C:\Users\User\host_image.png"
    watermark_image_path = r"C:\Users\User\watermark_image.png"
    
    # Load the host and watermark images
    host_image = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Embed the watermark
    watermarked_image = embed_watermark(host_image, watermark_image, alpha=20)
    cv2.imwrite(r"C:\Users\User\watermarked_image.png", watermarked_image)
    print("Watermark embedded and saved as 'watermarked_image.png' in User.")

    # Extract the watermark
    extracted_watermark = extract_watermark(host_image, watermarked_image, alpha=20)
    cv2.imwrite(r"C:\Users\User\extracted_watermark.jpg", extracted_watermark)
    print("Watermark extracted and saved as 'extracted_watermark.jpg' in User.")
