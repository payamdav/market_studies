import numpy as np

def zigzag(a, t, extendToEnd = False, start_point = None, direction = "left", percent = False):
  valt = lambda nv: nv * t if percent else t

  zz = np.zeros(len(a)).astype(int)
  if (start_point == None):
    start_point = np.argmax(a)
    nextType = 1
  else:
    nextType = 0
    if direction == "left":
      for i in range(start_point - 1, -1, -1):
        if a[i] > a[start_point] + valt(a[start_point]):
          nextType = -1
          break
        elif a[i] < a[start_point] - valt(a[start_point]):
          nextType = 1
          break
      if nextType == 0:
        return zz
    else:
      for i in range(start_point + 1, len(a)):
        if a[i] > a[start_point] + valt(a[start_point]):
          nextType = -1
          break
        elif a[i] < a[start_point] - valt(a[start_point]):
          nextType = 1
          break
      if nextType == 0:
        return zz
  zz[start_point] = nextType
  nextValue = a[start_point]
  nextIndex = None
  nextType = -1 * nextType
  for i in range(start_point + 1, len(a)):
    if nextType == -1:
      if not nextIndex and a[i] < nextValue - valt(nextValue):
        nextValue = a[i]
        nextIndex = i
      elif not nextIndex and a[i] > nextValue:
        nextValue = a[i]
      elif nextIndex and a[i] < nextValue:
        nextValue = a[i]
        nextIndex = i
      elif nextIndex and a[i] > nextValue + valt(nextValue):
        zz[nextIndex] = nextType
        nextValue = a[i]
        nextIndex = i
        nextType = -1 * nextType
    else:
      if not nextIndex and a[i] > nextValue + valt(nextValue):
        nextValue = a[i]
        nextIndex = i
      elif not nextIndex and a[i] < nextValue:
        nextValue = a[i]
      elif nextIndex and a[i] > nextValue:
        nextValue = a[i]
        nextIndex = i
      elif nextIndex and a[i] < nextValue - valt(nextValue):
        zz[nextIndex] = nextType
        nextValue = a[i]
        nextIndex = i
        nextType = -1 * nextType
  if nextIndex:
    zz[nextIndex] = nextType
    nextType = -1 * nextType
  if extendToEnd:
    zz[-1] = nextType
  nextValue = a[start_point]
  nextIndex = None
  nextType = -1
  for i in range(start_point - 1, -1, -1):
    if nextType == -1:
      if not nextIndex and a[i] < nextValue - valt(nextValue):
        nextValue = a[i]
        nextIndex = i
      elif not nextIndex and a[i] > nextValue:
        nextValue = a[i]
      elif nextIndex and a[i] < nextValue:
        nextValue = a[i]
        nextIndex = i
      elif nextIndex and a[i] > nextValue + valt(nextValue):
        zz[nextIndex] = nextType
        nextValue = a[i]
        nextIndex = i
        nextType = -1 * nextType
    else:
      if not nextIndex and a[i] > nextValue + valt(nextValue):
        nextValue = a[i]
        nextIndex = i
      elif not nextIndex and a[i] < nextValue:
        nextValue = a[i]
      elif nextIndex and a[i] > nextValue:
        nextValue = a[i]
        nextIndex = i
      elif nextIndex and a[i] < nextValue - valt(nextValue):
        zz[nextIndex] = nextType
        nextValue = a[i]
        nextIndex = i
        nextType = -1 * nextType
  if nextIndex:
    zz[nextIndex] = nextType
    nextType = -1 * nextType
  if extendToEnd:
    zz[0] = nextType
  return zz


