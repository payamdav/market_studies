from time import perf_counter

class TimerProfiler:
  def __init__(self, name="") -> None:
    self.checkpoints = []
    self.name = name
    self.checkpoint("start")

  def checkpoint(self, checkpoint_name=""):
    self.checkpoints.append((checkpoint_name, perf_counter()))
    if len(self.checkpoints) > 1:
      print(f">>> Timer: {self.name} - Checkpoint: {checkpoint_name} - Elapsed: {self.checkpoints[-1][1] - self.checkpoints[-2][1]}")
    else:
      print(f">>> Timer: {self.name} - Checkpoint: {checkpoint_name}")
    return self
  
  def report(self):
    print()
    print(f">>> >>> TimerProfiler: {self.name}")
    for i in range(len(self.checkpoints) - 1):
      print(f">>> {self.checkpoints[i + 1][0]}: {self.checkpoints[i + 1][1] - self.checkpoints[i][1]}")
    return self
