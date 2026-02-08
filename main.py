import torch
from math import prod as product

timeslots = [
	"09:15 - 10:30",
	"10:45 - 12:00",
]

locations = [
	"U01",
	"U02",
]

staff = [
	"Alex",
	"Andrew",
]

classes = [
	"Game Development",
	"Web Development",
]

shape = (len(timeslots), len(locations), len(staff), len(classes))

lessons = torch.rand(shape).unsqueeze(0)

neural_network = torch.nn.Sequential(
	torch.nn.Flatten(1),
	torch.nn.Linear(product(shape), product(shape)),
	torch.nn.Sigmoid(),
	torch.nn.Unflatten(1, shape),
)

optimiser = torch.optim.Adam(neural_network.parameters(), lr=0.01)

try:
	print("Press Ctrl+C to exit.")
	# Code this in a single-run mode, but anticipating a looping structure
	# while True:
	for _ in range(10):

		print("---")
		for (timeslot_id, timeslot) in enumerate(timeslots):
			for (location_id, location) in enumerate(locations):
				for (staff_id, staff_member) in enumerate(staff):
					for (class_id, class_name) in enumerate(classes):
						confidence = lessons[0, timeslot_id, location_id, staff_id, class_id].item()
						if confidence > 1/3:
							print(f"* {timeslot}, {location}, {staff_member}, {class_name}, {confidence:.2f}")

		lessons = neural_network(lessons)
		# optimiser.zero_grad()
		# Loss function for (location, timeslot) uniqueness
		# Loss function for (staff, timeslot) uniqueness
		# Loss function for (class, timeslot) uniqueness
		# loss.backward()
		# optimiser.step()
		lessons = lessons.detach()
except KeyboardInterrupt:
	pass

print("---")
