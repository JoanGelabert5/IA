from practica import joc
from typing import Optional, Tuple, Dict, Any
from .estat import Estat

class AgentMinimaxViatger(joc.Viatger):
    """
    Agente que implementa Minimax con poda alfa-beta, integrado con Viatger.
    Permite decidir la mejor acción en un tablero de agentes y destino.
    """

    # -------------------
    # Constantes de evaluación
    # -------------------
    VICTORIA = 1000           # Valor si el agente llega al destino
    DERROTA = -1000           # Valor si otro agente llega al destino
    BONUS_TURNO = 10          # Pequeña ventaja si es el turno del agente
    FACTOR_DISTANCIA = 10     # Factor de penalización según la distancia al destino

    def __init__(self, nom=None, n=5, m=5, profunditat_max=3):
        """
        Inicializa el agente Viatger con Minimax.
        
        Args:
            nom: nombre del agente (si None, se asigna automáticamente)
            n: ancho del tablero
            m: alto del tablero
            profunditat_max: profundidad máxima de búsqueda
        """
        super().__init__(nom=nom)
        self.n = n
        self.m = m
        self.profunditat_max = profunditat_max
        self._cache_estados = {}  # Cache para almacenar resultados de estados evaluados

    # -------------------
    # Método principal de acción
    # -------------------
    def actua(self, percepcio: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Devuelve la acción que debe realizar el agente en el turno actual.
        
        Args:
            percepcio: diccionario con información del estado actual del juego
        
        Returns:
            Una tupla (acción, dirección) o ("ESPERAR", None) si no se puede mover
        """
        estat_actual = self._construir_estat(percepcio)

        # Caso 1: el agente ya llegó al destino
        if estat_actual.agents.get(self.nom) == estat_actual.desti:
            return "ESPERAR", None

        # Caso 2: otro agente ya llegó al destino
        for nom_agente, pos in estat_actual.agents.items():
            if pos == estat_actual.desti:
                return "ESPERAR", None

        # Determina si el agente actual es maximizador en el turno
        es_maximizador = (estat_actual.torn == self.nom)

        # Ejecuta Minimax para obtener la mejor acción
        accio, _ = self.minimax(estat_actual, self.profunditat_max, float('-inf'), float('inf'), es_maximizador)

        # Si no se genera ninguna acción válida, esperar
        return accio if accio else ("ESPERAR", None)

    # -------------------
    # Construcción del estado
    # -------------------
    def _construir_estat(self, percepcio: Dict[str, Any]):
        """
        Convierte la percepción en un objeto de estado completo (Estat).
        """
        agents = percepcio["AGENTS"]
        parets = set(percepcio["PARETS"])
        desti = percepcio["DESTI"]
        torn = percepcio["TORN"]
        n, m = percepcio.get("MIDA", (self.n, self.m))
        return Estat(agents, parets, desti, torn, n, m)

    # -------------------
    # Algoritmo Minimax con poda alfa-beta
    # -------------------
    def minimax(self, estat, profunditat, alfa, beta, jugador_max):
        """
        Implementa Minimax con poda alfa-beta recursivo.
        """
        # Cache para evitar recalcular estados repetidos
        key = f"{hash(estat)}_{profunditat}_{jugador_max}"
        if key in self._cache_estados:
            return self._cache_estados[key]

        fills = estat.genera_fills()  # Genera todos los posibles movimientos
        # Caso base: profundidad 0, estado terminal o sin movimientos
        if profunditat == 0 or estat.es_terminal() or not fills:
            valor = self.valor(estat)
            return None, valor

        # Elegir función max o min según el turno
        if jugador_max:
            resultado = self._max_value(estat, fills, profunditat, alfa, beta)
        else:
            resultado = self._min_value(estat, fills, profunditat, alfa, beta)

        self._cache_estados[key] = resultado
        return resultado

    # -------------------
    # Funciones auxiliares de Minimax
    # -------------------
    def _max_value(self, estat, fills, profunditat, alfa, beta):
        """
        Evalúa los movimientos del jugador maximizador.
        """
        mejor_valor = float('-inf')
        mejor_accion = None
        for accio, direccio, fill, cost in fills:
            _, valor = self.minimax(fill, profunditat - 1, alfa, beta, False)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = (accio, direccio)
            alfa = max(alfa, mejor_valor)
            if alfa >= beta:
                break  # Poda beta
        return mejor_accion, mejor_valor

    def _min_value(self, estat, fills, profunditat, alfa, beta):
        """
        Evalúa los movimientos del jugador minimizador.
        """
        mejor_valor = float('inf')
        mejor_accion = None
        for accio, direccio, fill, cost in fills:
            _, valor = self.minimax(fill, profunditat - 1, alfa, beta, True)
            if valor < mejor_valor:
                mejor_valor = valor
                mejor_accion = (accio, direccio)
            beta = min(beta, mejor_valor)
            if alfa >= beta:
                break  # Poda alfa
        return mejor_accion, mejor_valor

    # -------------------
    # Función de evaluación heurística
    # -------------------
    def valor(self, estat):
        """
        Calcula el valor heurístico del estado para el agente.
        """
        # Caso victoria: el agente llegó al destino
        if estat.agents.get(self.nom) == estat.desti:
            return self.VICTORIA
        # Caso derrota: otro agente llegó al destino
        if any(pos == estat.desti for n, pos in estat.agents.items() if n != self.nom):
            return self.DERROTA

        # Evaluación basada en distancia Manhattan
        pos_actual = estat.agents.get(self.nom)
        otro_agente = next((n for n in estat.agents if n != self.nom and n != self.nom), None)

        dist_actual = abs(pos_actual[0]-estat.desti[0]) + abs(pos_actual[1]-estat.desti[1])
        ventaja = -dist_actual * self.FACTOR_DISTANCIA

        if otro_agente:
            dist_otro = abs(estat.agents[otro_agente][0]-estat.desti[0]) + abs(estat.agents[otro_agente][1]-estat.desti[1])
            ventaja += (dist_otro - dist_actual) * 5

        # Bonus si es el turno del agente
        if estat.torn == self.nom:
            ventaja += self.BONUS_TURNO

        return ventaja
