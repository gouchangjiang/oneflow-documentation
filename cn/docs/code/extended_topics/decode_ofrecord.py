import oneflow as flow
import oneflow.typing as oft
from typing import Tuple

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_train_config())
def train_job() -> Tuple[oft.Numpy,oft.Numpy]:
    images = flow.data.BlobConf("images",
                                shape=(28, 28, 1),
                                dtype=flow.float,
                                codec=flow.data.RawCodec())
    labels = flow.data.BlobConf("labels",
                                shape=(1, 1),
                                dtype=flow.int32,
                                codec=flow.data.RawCodec())

    return flow.data.decode_ofrecord("./dataset/", (images, labels),
                                     data_part_num=1,
                                     batch_size=3)


def main():
    check_point = flow.train.CheckPoint()
    check_point.init()

    f0, f1 = train_job()
    print(f0, f1)
    print(f0.shape, f1.shape)


if __name__ == '__main__':
    main()