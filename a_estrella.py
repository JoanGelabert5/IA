from practica import joc, agent
from collections import defaultdict
import math, queue

class Viatger_A_estrella(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger_A_estrella, self).__init__(*args, **kwargs)
        self.cami = []

    # Heurística: Distància Manhattan
    def heuristica(self, pos, desti):
        return abs(pos[0] - desti[0]) + abs(pos[1] - desti[1])

    #Generació dels successors d’un estat
    def genera_successors(self, pos, parets, mida):
        successors = []
        direccions = joc.Laberint.MOVS

        for direccio, (dx, dy) in direccions.items():
            #Aquí se construeix la jerarquia de moviments: BOTAR > MOURE > POSAR PARETS

            # BOTAR
            salt = (pos[0] + 2*dx, pos[1] + 2*dy)
            mig = (pos[0] + dx, pos[1] + dy)
            if (0 <= salt[0] < mida[0]) and (0 <= salt[1] < mida[1]) and \
               mig not in parets and salt not in parets:
                successors.append((salt, "BOTAR", direccio, 2))
                
            # MOURE
            nou = (pos[0] + dx, pos[1] + dy)
            if (0 <= nou[0] < mida[0]) and (0 <= nou[1] < mida[1]) and nou not in parets:
                successors.append((nou, "MOURE", direccio, 1))

            # POSAR PARET
            if (0 <= nou[0] < mida[0]) and (0 <= nou[1] < mida[1]) and nou not in parets:
                successors.append((pos, "POSAR PARET", direccio, 3))

        return successors

    # Acció principal de l’agent
    def actua(self, percepcio: dict):
        pos = percepcio["AGENTS"][self.nom]
        desti = percepcio["DESTI"]

        # Si ja som al destí
        if pos == desti:
            return ("ESPERAR", "")

        parets = set(percepcio["PARETS"])
        mida = percepcio["MIDA"]

        # Si ja tenim camí, el seguim
        if self.cami:
            return self.cami.pop(0)

        # Inicialització d’estructures per A* 
        #Un defaultdict es un diccionari que permet introduir de valor, a les claus buides, lo que un desitgi, en aquest cas un infinit
        g = defaultdict(lambda: math.inf)
        f = defaultdict(lambda: math.inf)
        pare = {}
        pq = queue.PriorityQueue()

        estat_inicial = pos
        g[estat_inicial] = 0
        f[estat_inicial] = self.heuristica(pos, desti)
        pq.put((f[estat_inicial], f[estat_inicial], estat_inicial))

        visitats = set()  # Conjunts per les caselles visitades

        while not pq.empty():
            _, _, actual = pq.get()

            # Si ja hem vist aquesta posició, la saltam
            if actual in visitats:
                continue
            visitats.add(actual)

            # Si arribem al destí, reconstruïm el camí
            if actual == desti:
                cami = []
                while actual in pare:
                    accio, direccio, anterior = pare[actual]
                    cami.append((accio, direccio))
                    actual = anterior
                cami.reverse()
                self.cami = cami
                return self.cami.pop(0) if self.cami else ("ESPERAR", "")

            # Generar successors
            for nou_pos, accio, direccio, cost in self.genera_successors(actual, parets, mida):
                nou_cost = g[actual] + cost
                if nou_cost < g[nou_pos]:
                    pare[nou_pos] = (accio, direccio, actual)
                    g[nou_pos] = nou_cost
                    h_nou = self.heuristica(nou_pos, desti)
                    f[nou_pos] = nou_cost + h_nou
                    pq.put((f[nou_pos], h_nou, nou_pos))

        return ("ESPERAR", "")
