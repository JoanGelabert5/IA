from practica import joc
from practica import agent
import heapq
import math


# ---------------------------------------------------------------------------
# Agent que implementa el BFS amb un deque
# ---------------------------------------------------------------------------
from collections import deque

class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)
        self.camí = []  # Guardarem el camí a seguir

    def actua(self, percepcio: dict):
        pos = percepcio["AGENTS"][self.nom]
        desti = percepcio["DESTI"]
        parets = set(percepcio["PARETS"])
        mida = percepcio["MIDA"]

        # Si ja tenim camí calculat, seguim-lo
        if self.camí:
            return "MOURE", self.camí.pop(0)

        # BFS per trobar el camí fins al destí
        movs = joc.Laberint.MOVS
        cua = deque([(pos, [])])  # cada element: (posició, camí_fins_ara)
        visitats = set({pos})

        while cua:
            actual, camí = cua.popleft()
            if actual == desti:
                self.camí = camí
                break #Hem trobat la meta

            for direccio, (dx, dy) in movs.items():
                nova_pos = (actual[0] + dx, actual[1] + dy)
                if (
                    0 <= nova_pos[0] < mida[0]
                    and 0 <= nova_pos[1] < mida[1]
                    and nova_pos not in parets
                    and nova_pos not in visitats #Evitar out of bounds
                ):
                    visitats.add(nova_pos) #Afegir al conjunt de visitats
                    cua.append((nova_pos, camí + [direccio]))

        # Si trobem camí, fem el primer pas
        if self.camí:
            return "MOURE", self.camí.pop(0) #Fins que no se buidi el camí, mos mourem amb el que treguem de self.cami -> (N,O,S,E)
        else:
            return "ESPERAR", None

