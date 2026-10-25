class Estat:
    """
    Classe que representa l'estat del joc del laberint.
    """

    # Direccions possibles per moure's
    MOVS = {
        "N": (0, -1),
        "S": (0, 1),
        "E": (1, 0),
        "O": (-1, 0)
    }

    def __init__(self, agents, parets, desti, torn, n, m):
        """
        Inicialitza un estat del joc.

        Args:
            agents (dict): Diccionari amb noms i posicions dels agents
            parets (set): Conjunt de tuples (x,y) amb posicions de parets
            desti (tuple): Posició destí (x,y)
            torn (str): Nom de l'agent que té el torn
            n (int): Amplada del taulell
            m (int): Alçada del taulell
        """
        self.agents = agents.copy()
        self.parets = set(parets)
        self.desti = desti
        self.torn = torn
        self.n = n
        self.m = m

    def __eq__(self, other):
        """Compara dos estats per igualtat."""
        if not isinstance(other, Estat):
            return False
        return (self.agents == other.agents and
                self.parets == other.parets and
                self.desti == other.desti and
                self.torn == other.torn and
                self.n == other.n and
                self.m == other.m)

    def __hash__(self):
        """Permet utilitzar l'estat en sets i com a clau de diccionaris."""
        agents_frozen = frozenset((k, tuple(v)) for k, v in self.agents.items())
        parets_frozen = frozenset(self.parets)
        return hash((agents_frozen, parets_frozen, self.desti, self.torn, self.n, self.m))

    def es_meta(self):
        """Retorna True si l'agent del torn actual ha arribat al destí."""
        return self.agents[self.torn] == self.desti

    def es_terminal(self):
        """
        Determina si l'estat és terminal:
        - Algún agent ha arribat al destí
        - No hi ha moviments possibles per l'agent actual
        """
        if any(pos == self.desti for pos in self.agents.values()):
            return True
        return len(self.genera_fills()) == 0

    def _torn_seguent(self):
        """Retorna el nom del següent agent en torn."""
        agents_list = list(self.agents.keys())
        current_index = agents_list.index(self.torn)
        next_index = (current_index + 1) % len(agents_list)
        return agents_list[next_index]

    def _posicio_valida(self, pos):
        """Comprova si una posició està dins del taulell."""
        x, y = pos
        return 0 <= x < self.n and 0 <= y < self.m

    def _es_accessible(self, pos):
        """
        Verifica si una posició és accessible:
        - Dins del taulell
        - Sense paret
        - Sense un altre agent
        """
        return self._posicio_valida(pos) and pos not in self.parets and pos not in self.agents.values()

    def genera_fills(self):
        """
        Genera tots els estats successors possibles per l'agent actual.

        Accions possibles:
            - "MOURE": Mou-se a una casella adjacent (cost 1)
            - "BOTAR": Salta dues caselles (cost 2)
            - "POSAR_PARET": Col·loca una paret adjacent (cost 3)

        Returns:
            list: Llista de tuples (accio, direccio, nou_estat, cost)
        """
        fills = []
        pos_actual = self.agents[self.torn]

        # --------------------
        # Acció MOURE: moure's a casella adjacent
        # --------------------
        for direccio, mov in self.MOVS.items():
            nova_pos = (pos_actual[0] + mov[0], pos_actual[1] + mov[1])
            if self._es_accessible(nova_pos):
                nou_agents = self.agents.copy()
                nou_agents[self.torn] = nova_pos
                nou_parets = set(self.parets)
                nou_parets.add(pos_actual)  # bloquejar posició anterior
                nou_torn = self._torn_seguent()
                nou_estat = Estat(nou_agents, nou_parets, self.desti, nou_torn, self.n, self.m)
                fills.append(("MOURE", direccio, nou_estat, 1))

        # --------------------
        # Acció BOTAR: saltar dues caselles
        # --------------------
        for direccio, mov in self.MOVS.items():
            pos_intermedia = (pos_actual[0] + mov[0], pos_actual[1] + mov[1])
            nova_pos = (pos_actual[0] + 2 * mov[0], pos_actual[1] + 2 * mov[1])
            if self._posicio_valida(pos_intermedia) and pos_intermedia not in self.parets and self._es_accessible(nova_pos):
                nou_agents = self.agents.copy()
                nou_agents[self.torn] = nova_pos
                nou_parets = set(self.parets)
                nou_parets.add(pos_actual)
                nou_torn = self._torn_seguent()
                nou_estat = Estat(nou_agents, nou_parets, self.desti, nou_torn, self.n, self.m)
                fills.append(("BOTAR", direccio, nou_estat, 2))

        # --------------------
        # Acció POSAR_PARET: posar paret adjacent
        # --------------------
        for direccio, mov in self.MOVS.items():
            pos_paret = (pos_actual[0] + mov[0], pos_actual[1] + mov[1])
            if (self._posicio_valida(pos_paret) and pos_paret != self.desti
                and pos_paret not in self.agents.values() and pos_paret not in self.parets):
                nou_agents = self.agents.copy()
                nou_parets = self.parets | {pos_paret}
                nou_torn = self._torn_seguent()
                nou_estat = Estat(nou_agents, nou_parets, self.desti, nou_torn, self.n, self.m)
                fills.append(("POSAR_PARET", direccio, nou_estat, 3))

        return fills

    def heuristica(self):
        """
        Heurística simple: distància Manhattan de l'agent del torn actual fins al destí.

        Returns:
            int: distància Manhattan
        """
        pos_actual = self.agents[self.torn]
        return abs(pos_actual[0] - self.desti[0]) + abs(pos_actual[1] - self.desti[1])
