import cv2
import numpy as np


def encode_image(image_path, secret_data, output_path):
    image = cv2.imread(image_path)
    secret_data += "####"  # סימון סוף הודעה
    binary_secret_data = ''.join([format(ord(i), "08b") for i in secret_data])

    data_index = 0
    for row in image:
        for pixel in row:
            for channel in range(3):
                if data_index < len(binary_secret_data):
                    pixel[channel] = int(format(pixel[channel], "08b")[:-1] + binary_secret_data[data_index], 2)
                    data_index += 1
    cv2.imwrite(output_path, image)
    return "החתימה הוטמעה בהצלחה!"


def decode_image(image_path):
    image = cv2.imread(image_path)
    binary_data = ""
    for row in image:
        for pixel in row:
            for channel in range(3):
                binary_data += format(pixel[channel], "08b")[-1]

    all_bytes = [binary_data[i: i + 8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data.endswith("####"):
            return decoded_data[:-4]
    return "לא נמצאה חתימה."