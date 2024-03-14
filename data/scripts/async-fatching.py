import asyncio
import os
from typing import List

import aiohttp
import pandas as pd
from aiohttp import ClientResponseError
from load import parse_image_content, save_cvimage
from tqdm import tqdm

fetched = []


class ImageCollector:
    def __init__(self, urls: List[str] = None) -> None:
        self.urls = urls
        self.valid_urls = []
        self.images = []

    @classmethod
    async def fetch_image(cls, url: str, session: aiohttp.ClientSession) -> bytes:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()
        except ClientResponseError:
            return None

    @classmethod
    async def is_valid_url(cls, url: str, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return True
        except ClientResponseError:
            return False

    @classmethod
    async def get_image_from_links(cls, urls: List[str]) -> List[bytes]:
        async with aiohttp.ClientSession() as session:
            tasks = [ImageCollector.fetch_image(url, session) for url in urls]
            return await asyncio.gather(*tasks)

    @classmethod
    async def test_urls(cls, urls: List[str]) -> List[bytes]:
        async with aiohttp.ClientSession() as session:
            tasks = [ImageCollector.is_valid_url(url, session) for url in urls]
            return await asyncio.gather(*tasks)

    async def run(self, urls: List[str], check_url: bool = False):
        try:
            if not check_url:
                images = await ImageCollector.get_image_from_links(urls)
                self.images.extend(images)
            else:
                self.valid_urls.extend(await ImageCollector.test_urls(urls))

        except ValueError:
            pass


def chunk_list(lst, chunk_size):

    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def persist_images(images: List[bytes], dirs: List[str]) -> bool:
    for image, dir in zip(images, dirs):
        img = parse_image_content(image)
        result = save_cvimage(img, dir)
        if not result:
            print(f"Faild To Persist {os.path.basename(dir)}")


async def main():
    BASE_TO_SAVE = "data/present/modelling/v1/"
    df = pd.read_csv("data/processed/v4/all-products.csv")
    urls = df["image"].to_list()
    tags = df["tag"].to_list()
    chunk_size = 512
    # collected_data = []
    length = len(urls) // chunk_size
    chunks = chunk_list(urls, chunk_size=chunk_size)
    # tag_chunks = chunk_list(tags,chunk_size)
    images_binary = []
    # assert len(tag_chunks)==len(chunks)
    for chunk_urls in tqdm(chunks, desc="Process URLs", total=length):
        collector = ImageCollector()
        await collector.run(chunk_urls, check_url=False)
        # paths = [BASE_TO_SAVE+name+'/'+str(idx) for name in tag_chunks]
        # persist_images(collector.images,paths)
        images_binary.extend(collector.images)
        if len(images_binary) % 1024:
            print(f"Image Collected:{len(images_binary)}")
    paths = [BASE_TO_SAVE + name + "/" + str(idx) for idx, name in enumerate(tags)]
    persist_images(images_binary, paths)


if __name__ == "__main__":
    asyncio.run(main())

    # df['valid_image'] = collected_data
    # df.to_csv('data/processed/v3/all-products.csv')
