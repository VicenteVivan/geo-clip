import torch
from PIL import Image
from torch import nn

from geoclip.model.GeoCLIP import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader
from geoclip.train.train import train


def test_gps_queue_accepts_batches_that_do_not_divide_queue_size():
    model = GeoCLIP.__new__(GeoCLIP)
    nn.Module.__init__(model)
    model._initialize_gps_queue(queue_size=5)

    first_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    second_batch = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    model.dequeue_and_enqueue(first_batch)
    model.dequeue_and_enqueue(second_batch)

    assert int(model.gps_queue_ptr) == 1
    assert torch.equal(model.gps_queue[:, 0], second_batch[-1])
    assert torch.equal(model.gps_queue[:, 3:].t(), second_batch[:2])


def test_dataloader_returns_tensor_coordinates(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2)).save(image_path)

    dataset = GeoDataLoader.__new__(GeoDataLoader)
    dataset.images = [str(image_path)]
    dataset.coordinates = [(28.5383, -81.3792)]
    dataset.transform = None

    _, coordinates = dataset[0]

    assert coordinates.dtype == torch.float32
    assert coordinates.shape == (2,)


def test_training_accepts_a_short_final_batch():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("gps_queue", torch.zeros(2, 2))

        def get_gps_queue(self):
            return self.gps_queue.t()

        def dequeue_and_enqueue(self, gps):
            self.gps_queue[:, : gps.shape[0]] = gps.t()

        def forward(self, images, locations):
            return self.scale * torch.ones(images.shape[0], locations.shape[0])

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        (torch.zeros(2, 1), torch.zeros(2, 2)),
        (torch.zeros(1, 1), torch.zeros(1, 2)),
    ]

    train(batches, model, optimizer, epoch=1, batch_size=2, device="cpu")
