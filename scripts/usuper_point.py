import pyuTensor as ut
import numpy as np
import torch
import pyuTensor
import tensorflow as tf
from superpoint_tf import SuperPointTF, NMSPool

class ConvLayer:
    def __init__(self, weight, bias, padding="SAME", strides=None):
        self.weight = weight
        self.bias = bias
        self.padding = padding
        self.strides = strides or [1, 1, 1, 1]

    def __call__(self, x):
        return ut.conv2d_f(x, kernel=self.weight, bias=self.bias, padding=self.padding, strides=self.strides)

class uSuperPoint:
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
    }
    
    def __init__(self, state_dict_path: str, config):
        self.config = {**self.default_config, **config}
        self._state_dict = {
            k: v.detach().cpu().numpy() for k, v in torch.load(state_dict_path).items()
        }
        for k, v in self._state_dict.items():
            if k.endswith(".weight"):
                self._state_dict[k] = v.transpose([0, 2, 3, 1])
        
        self._nms_pool = NMSPool(self.config["nms_radius"])
        
        # Set up uTensor memory usage
        pyuTensor.set_ram_total(1024*1000*1000)
        pyuTensor.set_meta_total(1024*1000)

    def __call__(self, img: np.ndarray):
        x = self.call_encoder(img)
        descripter = self.call_dense_descriptor(x)
        scores = self.call_scores(x)
        return x, scores, descripter
    
    def call_(self, img: np.ndarray):
        _, scores, dense_descriptors = self(img)
        scores = tf.convert_to_tensor(scores)
        dense_descriptors = tf.convert_to_tensor(dense_descriptors)
        print(scores.shape)
        print(dense_descriptors.shape)
        
        # Discard keypoints near the image borders
        # TODO: check scores dimension
        _, height, width = scores.shape
        keypoints = [tf.where(s > self.config["keypoint_threshold"]) for s in scores]
        scores = [tf.gather_nd(s, indices=k) for s, k in zip(scores, keypoints)]
        keypoints, scores = list(
            zip(
                *[
                    self.remove_borders(k, s, height, width)
                    for k, s in zip(keypoints, scores)
                ]
            )
        )
        # Keep the k keypoints with highest score
        keypoints, scores = list(
            zip(*[self.top_k_keypoints(k, s) for k, s in zip(keypoints, scores)])
        )
        # Convert (h, w) to (x, y)
        keypoints = [tf.cast(k[:, ::-1], dtype=tf.float32) for k in keypoints]
        # Extract descriptors
        descriptors = [
            self.sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, dense_descriptors)
        ]
        return keypoints, scores, descriptors
    
    def call_encoder(self, img: np.ndarray):
        x = pyuTensor.conv2d(input=img, kernel=self._state_dict["conv1a.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv1a.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv1b.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv1b.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.max_pool2d(x, [2, 2], [1, 2, 2, 1], padding="SAME")
        
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv2a.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv2a.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv2b.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv2b.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.max_pool2d(x, [2, 2], [1, 2, 2, 1], padding="SAME")
        
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv3a.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv3a.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv3b.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv3b.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.max_pool2d(x, [2, 2], [1, 2, 2, 1], padding="SAME")
        
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv4a.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv4a.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["conv4b.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["conv4b.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        
        return x
    
    def call_dense_descriptor(self, x):
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["convDa.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["convDa.bias"], padding="SAME")
        x = pyuTensor.relu(x)
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["convDb.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["convDb.bias"], padding="SAME")
        
        x = tf.convert_to_tensor(x) # change numpy array to tensor
        x = tf.linalg.normalize(x, ord=2, axis=-1)[0]
        x = x.numpy()
        return x
    
    def call_scores(self, x):
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["convPa.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["convPa.bias"], padding="SAME")
        x = pyuTensor.relu(x) # cPa
        x = pyuTensor.conv2d(input=x, kernel=self._state_dict["convPb.weight"], strides=[1, 1, 1, 1], bias=self._state_dict["convPb.bias"], padding="SAME")
        
        x = tf.convert_to_tensor(x) # change numpy array to tensor
        scores = tf.nn.softmax(x, axis=-1)[:, :, :, :-1]
        b, h, w, _ = scores.shape
        scores = tf.reshape(scores, (b, h, w, 8, 8))
        scores = tf.reshape(
            tf.transpose(scores, perm=[0, 1, 3, 2, 4]), (b, h * 8, w * 8)
        )
        x = self.simple_nms(scores)
        x.numpy()
        return x

    def simple_nms(self, scores):
        zeros = tf.zeros_like(scores)
        max_mask = scores == self._nms_pool(scores)
        for _ in range(2):
            supp_mask = self._nms_pool(tf.cast(max_mask, dtype=tf.float32)) > 0
            supp_scores = tf.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == self._nms_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return tf.where(max_mask, scores, zeros)
    
    def remove_borders(self, keypoints, scores, height, width):
        """Removes keypoints too close to the border"""
        border = self.config["remove_borders"]
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]
    
    def top_k_keypoints(self, keypoints, scores):
        k = self.config["max_keypoints"]
        if k >= len(keypoints) or k <= 0:
            return keypoints, scores
        scores, indices = tf.math.top_k(scores, k)
        return tf.gather(keypoints, indices), scores
    
    @staticmethod
    def sample_descriptors(keypoints: tf.Tensor, descriptors: tf.Tensor, s: int = 8):
        import torch

        # TODO: pure tensorflow grid_sampl impl
        # https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
        # https://stackoverflow.com/questions/52888146/what-is-the-equivalent-of-torch-nn-functional-grid-sample-in-tensorflow-numpy
        """Interpolate descriptors at keypoint locations"""
        descriptors = tf.transpose(descriptors, [0, 3, 1, 2])
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5  # BxNx2
        keypoints /= tf.convert_to_tensor(
            [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)], dtype=keypoints.dtype
        )[
            tf.newaxis
        ]  # BxNx2
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        th_descriptors = torch.from_numpy(descriptors.numpy())
        th_keypoints = torch.from_numpy(keypoints.numpy())
        args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
        th_descriptors = torch.nn.functional.grid_sample(
            th_descriptors, th_keypoints.view(b, 1, -1, 2), mode="bilinear", **args
        )
        th_descriptors = torch.nn.functional.normalize(
            th_descriptors.reshape(b, c, -1), p=2, dim=1
        )
        return tf.transpose(
            tf.convert_to_tensor(th_descriptors.detach().cpu().numpy()), [0, 2, 1]
        )

    
if __name__ == '__main__':
    import cv2
    import torch
    from superpoint_tf import SuperPointTF

    # read image
    img = cv2.resize(cv2.imread("imgs/rabbit.jpeg"), dsize=None, fx=0.5, fy=0.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.float32)
    # change img to 4 dim
    img = np.expand_dims(img, axis=0)
    superpoint_weights_file = 'scripts/superpoint_v1.pth'
    
    print(img.shape)
    
    config = {
        "nms_radius": 2, # args.nms_radius,
        "keypoint_threshold": 0.005, # args.keypoint_threshold,
        "max_keypoints": 1000, # args.max_keypoints,
    }
    
    # test model
    model = uSuperPoint(state_dict_path=superpoint_weights_file, config=config)
    
    # ground truth model
    superpoint_tf = SuperPointTF(config)
    superpoint_tf.load_torch_state_dict(superpoint_weights_file)
    
    gt_keypoints, gt_keypoints_scores, gt_descriptors = superpoint_tf.call_(img)
    
    keypoints, scores, descriptors = model.call_(img)
