import os
import argparse
from icrawler.builtin import GoogleImageCrawler

def scrape_images(keyword, max_num, output_dir):
    """
    Scrapes images from Google Images based on a keyword.
    Used by the Admin Dataset Refresh Tool to augment the training set.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scraping up to {max_num} images for keyword: '{keyword}' into {output_dir}")
    
    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        storage={'root_dir': output_dir}
    )
    
    # filter for faces / photos roughly
    filters = dict(
        type='photo',
        color='color'
    )
    
    google_crawler.crawl(
        keyword=keyword, 
        filters=filters, 
        max_num=max_num, 
        file_idx_offset=0
    )
    print("Scraping completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Admin Dataset Collection Scraper")
    parser.add_argument('--keyword', type=str, required=True, help="Search keyword (e.g. 'AI generated realistic faces')")
    parser.add_argument('--max', type=int, default=50, dest='max_num', help="Max number of images to download")
    parser.add_argument('--output', type=str, default='data/train/fake', help="Output directory path")
    
    args = parser.parse_args()
    scrape_images(args.keyword, args.max_num, args.output)
