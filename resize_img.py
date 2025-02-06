from PIL import Image

image = Image.open("images/000082.jpg")  # 画像を開く
resized_image = image.resize((64, 64))  # 64x64にリサイズ
resized_image.save("resize_000082.jpg")  # 保存