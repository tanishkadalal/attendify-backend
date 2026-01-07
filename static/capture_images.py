# import cv2
# import os

# name = "Ishu"                     # your name folder
# save_dir = f"data/{name}"
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)

# cap = cv2.VideoCapture(0)
# count = 0
# total_images = 40                 # number of images to capture

# print("✅ Starting image capture...")
# print("Look at the camera — normal, smiling, left, right, up, down")

# while count < total_images:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     cv2.imshow("Capturing Images", frame)

#     # Save image every 3 frames
#     if count < total_images:
#         img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
#         cv2.imwrite(img_path, frame)
#         count += 1
#         print(f"Saved: {img_path}")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# print("✅ Capture complete! Images saved in:", save_dir)
# cap.release()
# cv2.destroyAllWindows()


import cv2
import os

name = "Tanish 22501091"
save_dir = f"data/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture a photo, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)
    
    if key == ord(' '):  # SPACE key
        img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
