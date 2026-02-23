"""
CPU Manager para procesamiento de embeddings con CPU.

Gestiona recursos de CPU y RAM, monitorea uso, y proporciona
estadísticas en tiempo real. Implementa singleton pattern para
asegurar una única instancia de gestión.

Basado en psutil 6.1.1 para monitoreo cross-platform.
"""
import gc
import logging
import psutil
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ResourceStats:
    """
    Estadísticas de recursos del sistema.
    
    Attributes:
        cpu_percent: Porcentaje de uso de CPU
        cpu_count_physical: Número de cores físicos
        cpu_count_logical: Número de cores lógicos
        cpu_freq_current: Frecuencia actual de CPU en MHz
        ram_total_gb: RAM total en GB
        ram_used_gb: RAM usada en GB
        ram_available_gb: RAM disponible en GB
        ram_percent: Porcentaje de RAM usada
        inference_count: Número de inferencias realizadas
    """
    cpu_percent: float
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_freq_current: Optional[float]
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    inference_count: int


class CPUManager:
    """
    Gestor de recursos de CPU para procesamiento de embeddings.
    
    Singleton que maneja:
    - Configuración de hilos de PyTorch
    - Monitoreo de CPU y RAM
    - Logging periódico de recursos
    - Limpieza de memoria (garbage collection)
    
    Example:
        ```python
        from services.cpu_manager import get_cpu_manager
        
        cpu_mgr = get_cpu_manager(num_threads=4)
        cpu_mgr.log_usage()
        cpu_mgr.track_inference()
        ```
    """
    
    _instance: Optional['CPUManager'] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Implementa singleton pattern.
        
        Asegura que solo exista una instancia del gestor.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        num_threads: Optional[int] = None,
        log_every_n: int = 10,
        log_enabled: bool = True
    ):
        """
        Inicializa el gestor de CPU.
        
        Args:
            num_threads: Número de hilos a usar (None = auto-detect)
            log_every_n: Log cada N inferencias
            log_enabled: Habilitar logging automático
        """
        # Evitar re-inicialización
        if hasattr(self, '_initialized'):
            return
        
        # Configuración
        self.log_every_n = log_every_n
        self.log_enabled = log_enabled
        self.device = "cpu"
        
        # Contadores
        self._inference_count = 0
        
        # Detección de CPU
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        
        # Configurar hilos de PyTorch
        if num_threads is None:
            # Auto-detect: usar cores lógicos
            self.num_threads = self.cpu_count_logical
        else:
            # Validar que no exceda cores disponibles
            self.num_threads = min(num_threads, self.cpu_count_logical)
        
        # Aplicar configuración de hilos a PyTorch
        torch.set_num_threads(self.num_threads)
        
        # Log inicial
        self._log_initialization()
        
        self._initialized = True
    
    def _log_initialization(self) -> None:
        """Log de información inicial del sistema."""
        try:
            # Obtener frecuencia de CPU si está disponible
            cpu_freq = psutil.cpu_freq()
            freq_str = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A"
            
            # RAM total
            mem = psutil.virtual_memory()
            ram_total_gb = mem.total / (1024**3)
            
            logger.info("=" * 70)
            logger.info("💻 CPU MANAGER INICIALIZADO")
            logger.info(f"   Dispositivo: CPU")
            logger.info(f"   Cores físicos: {self.cpu_count_physical}")
            logger.info(f"   Cores lógicos (threads): {self.cpu_count_logical}")
            logger.info(f"   Hilos PyTorch configurados: {self.num_threads}")
            logger.info(f"   Frecuencia CPU: {freq_str}")
            logger.info(f"   RAM Total: {ram_total_gb:.2f} GB")
            logger.info(f"   Logging cada {self.log_every_n} inferencias")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.warning(f"Error al obtener info inicial del sistema: {e}")
    
    def get_cpu_info(self) -> Dict[str, any]:
        """
        Obtiene información detallada de la CPU.
        
        Returns:
            Dict con información de CPU
        """
        try:
            cpu_freq = psutil.cpu_freq()
            
            return {
                "physical_cores": self.cpu_count_physical,
                "logical_cores": self.cpu_count_logical,
                "pytorch_threads": self.num_threads,
                "current_freq_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "min_freq_mhz": round(cpu_freq.min, 2) if cpu_freq else None,
                "max_freq_mhz": round(cpu_freq.max, 2) if cpu_freq else None,
            }
        except Exception as e:
            logger.warning(f"Error al obtener info de CPU: {e}")
            return {
                "physical_cores": self.cpu_count_physical,
                "logical_cores": self.cpu_count_logical,
                "pytorch_threads": self.num_threads,
            }
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtiene información de memoria RAM.
        
        Returns:
            Dict con información de memoria en GB y porcentajes
        """
        try:
            mem = psutil.virtual_memory()
            
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": round(mem.percent, 1),
                "free_gb": round(mem.free / (1024**3), 2),
            }
        except Exception as e:
            logger.error(f"Error al obtener info de memoria: {e}")
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "available_gb": 0.0,
                "percent": 0.0,
                "free_gb": 0.0,
            }
    
    def get_cpu_usage(self, interval: float = 1.0) -> Tuple[float, Optional[list]]:
        """
        Obtiene uso actual de CPU.
        
        Args:
            interval: Tiempo de medición en segundos
                      interval=1.0 → medición precisa pero bloquea 1s
                      interval=None → no bloquea pero menos preciso
        
        Returns:
            Tupla (uso_total, uso_por_core)
        """
        try:
            # Uso total (bloquea por interval segundos para precisión)
            total = psutil.cpu_percent(interval=interval)
            
            # Uso por core (no vuelve a bloquear)
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            
            return round(total, 1), [round(x, 1) for x in per_cpu]
            
        except Exception as e:
            logger.warning(f"Error al obtener uso de CPU: {e}")
            return 0.0, None
    
    def get_stats(self, include_cpu_usage: bool = False) -> ResourceStats:
        """
        Obtiene estadísticas completas de recursos.
        
        Args:
            include_cpu_usage: Si True, incluye medición de CPU
                               (bloquea ~1s para precisión)
        
        Returns:
            ResourceStats con todas las métricas
        """
        # CPU usage (opcional porque bloquea)
        if include_cpu_usage:
            cpu_percent, _ = self.get_cpu_usage(interval=1.0)
        else:
            # Medición rápida (no bloquea)
            cpu_percent = round(psutil.cpu_percent(interval=None), 1)
        
        # Frecuencia de CPU
        try:
            cpu_freq = psutil.cpu_freq()
            freq = round(cpu_freq.current, 2) if cpu_freq else None
        except:
            freq = None
        
        # Memoria
        mem_info = self.get_memory_info()
        
        return ResourceStats(
            cpu_percent=cpu_percent,
            cpu_count_physical=self.cpu_count_physical,
            cpu_count_logical=self.cpu_count_logical,
            cpu_freq_current=freq,
            ram_total_gb=mem_info["total_gb"],
            ram_used_gb=mem_info["used_gb"],
            ram_available_gb=mem_info["available_gb"],
            ram_percent=mem_info["percent"],
            inference_count=self._inference_count,
        )
    
    def log_usage(self, force: bool = False) -> None:
        """
        Log de uso de recursos con formato legible.
        
        Args:
            force: Forzar log incluso si logging está deshabilitado
        """
        if not self.log_enabled and not force:
            return
        
        try:
            # No bloquear para log rápido
            cpu_percent = round(psutil.cpu_percent(interval=None), 1)
            mem = psutil.virtual_memory()
            
            # Emoji basado en uso
            if cpu_percent < 50:
                cpu_emoji = "🟢"
            elif cpu_percent < 80:
                cpu_emoji = "🟡"
            else:
                cpu_emoji = "🔴"
            
            if mem.percent < 70:
                ram_emoji = "🟢"
            elif mem.percent < 90:
                ram_emoji = "🟡"
            else:
                ram_emoji = "🔴"
            
            logger.info(
                f"{cpu_emoji} CPU: {cpu_percent}% | "
                f"{ram_emoji} RAM: {mem.used / (1024**3):.2f}/{mem.total / (1024**3):.2f} GB "
                f"({mem.percent:.1f}%)"
            )
            
        except Exception as e:
            logger.warning(f"Error al log de recursos: {e}")
    
    def track_inference(self) -> None:
        """
        Registra una inferencia y log periódico automático.
        
        Incrementa contador y hace log cada N inferencias
        según configuración de log_every_n.
        """
        self._inference_count += 1
        
        # Log automático periódico
        if self._inference_count % self.log_every_n == 0:
            self.log_usage()
    
    def clear_cache(self) -> None:
        """
        Limpia cache de Python (garbage collection).
        
        Nota: En CPU no hay cache de GPU que limpiar,
        pero sí podemos forzar garbage collection de Python.
        """
        try:
            collected = gc.collect()
            logger.debug(f"🧹 Garbage collection: {collected} objetos limpiados")
        except Exception as e:
            logger.warning(f"Error en garbage collection: {e}")
    
    def check_memory_available(self, required_gb: float = 1.0) -> bool:
        """
        Verifica si hay suficiente RAM disponible.
        
        Args:
            required_gb: GB de RAM requeridos
        
        Returns:
            True si hay suficiente RAM disponible
        """
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            
            if available_gb < required_gb:
                logger.warning(
                    f"⚠️ RAM insuficiente: {available_gb:.2f}GB disponible, "
                    f"{required_gb:.2f}GB requeridos"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error al verificar memoria: {e}")
            return True  # Asumir que hay memoria si falla la verificación
    
    def get_inference_count(self) -> int:
        """Retorna el contador de inferencias."""
        return self._inference_count
    
    def reset_inference_count(self) -> None:
        """Reinicia el contador de inferencias."""
        self._inference_count = 0
        logger.debug("Contador de inferencias reiniciado")


# ===================================
# SINGLETON GLOBAL
# ===================================

_cpu_manager_instance: Optional[CPUManager] = None


def get_cpu_manager(
    num_threads: Optional[int] = None,
    log_every_n: int = 10,
    log_enabled: bool = True
) -> CPUManager:
    """
    Obtiene instancia única del gestor de CPU.
    
    Args:
        num_threads: Número de hilos (None = auto-detect)
        log_every_n: Log cada N inferencias
        log_enabled: Habilitar logging automático
    
    Returns:
        Instancia singleton de CPUManager
    
    Example:
        ```python
        # Primera llamada: crea instancia
        cpu_mgr = get_cpu_manager(num_threads=4)
        
        # Siguientes llamadas: retorna misma instancia
        cpu_mgr2 = get_cpu_manager()  # Misma instancia
        assert cpu_mgr is cpu_mgr2  # True
        ```
    """
    global _cpu_manager_instance
    
    if _cpu_manager_instance is None:
        _cpu_manager_instance = CPUManager(
            num_threads=num_threads,
            log_every_n=log_every_n,
            log_enabled=log_enabled
        )
    
    return _cpu_manager_instance