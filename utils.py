import requests

def download_image(path, image_url, image_id):
  response = requests.get(image_url)
  img = response.content
  if f'{image_id}.jpg' not in os.listdir(path):
      with open(path + "/" + f'{image_id}' + '.jpg', 'wb') as handler:
        handler.write(img)


def load_images_to_dir(path, imageset):
  # load images concurrently
  with concurrent.futures.ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        future_to_url = {
            executor.submit(download_image, path, image.file_name, image_id): image.file_name
            for image_id, image in imageset.images.items()
        }
        for future in tqdm(concurrent.futures.as_completed(
            future_to_url
        )):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    "%r generated an exception: %s" % (url, exc)
                )


def get_intermediate_dict(data):
  result = {}
  for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    if image_id in result:
      result[image_id].add_annotation(annotation["caption"])
    else:
      result[image_id] = Image(image_id, [annotation["caption"]])
  for image in data["images"]:
    if image['id'] not in result:
      continue
    else:
      result[image['id']].file_name = image['flickr_url']
  return result