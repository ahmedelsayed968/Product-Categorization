import cv2
import numpy as np
import requests
from requests import Response


# from cv2 import MatLike
def get_image_from_link(url: str):
    try:
        response: Response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return parse_image_content(response.content)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch image from {url}: {e}")


def parse_image_content(image_content: bytes) -> np.array:
    image_array = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    return image_array


def save_cvimage(image: np.array, file_name: str, extension="jpg") -> bool:
    res = cv2.imwrite(f"{file_name}.{extension}", image)
    return res


if __name__ == "__main__":
    # print(parse_image_content(get_image_from_link('https://m.media-amazon.com/images/I/71eUwDk8z+L._AC_UL320_.jpg')))
    pass
