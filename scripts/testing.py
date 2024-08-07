from pathlib import Path
from superpoint_tf import SuperPointTF
import _pyuSuperpoint
import cv2

# prepare model
sp = _pyuSuperpoint.Superpoint()
sp_tf = SuperPointTF(
    {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 1000,
    }
)

# prepare image
img_path = Path(__file__).parent.parent / "cifar_cnn" / "imgs" / "0.png"
img = cv2.imread(str(img_path.absolute()), cv2.IMREAD_GRAYSCALE)
img = img.reshape(1, 32, 32, 1)
img = img.astype("float32") / 255.0
print(f"Image shape: {img.shape}")

# run detection
print(f"Running inference for SuperPoint (TF):")
out_tf = sp_tf.call_encoder(img)
print("Running inference for SuperPoint (C++):")
out = sp.inference(img)

# print keypoints
print(f"{out.shape=}")
print(f"{out=}")

print(f"{out_tf.shape=}")
print(f"{out_tf=}")
