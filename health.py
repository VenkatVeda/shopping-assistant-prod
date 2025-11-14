"""
Health monitoring module for the Shopping Assistant application.
Provides health check functionality for Docker containers and system monitoring.
"""

import json
import time
import psutil
from typing import Dict, Any
from datetime import datetime


class HealthChecker:
    """Monitors application health and system resources"""
    
    def __init__(self, app_instance):
        """Initialize health checker with application instance"""
        self.app = app_instance
        self.start_time = time.time()
        self.last_check = None
        self.status = "healthy"
        
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_check = datetime.now().isoformat()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        
        # Determine overall health status
        status = "healthy"
        warnings = []
        
        # Check CPU usage
        if cpu_percent > 90:
            status = "degraded"
            warnings.append(f"High CPU usage: {cpu_percent}%")
        
        # Check memory usage
        if memory.percent > 90:
            status = "degraded"
            warnings.append(f"High memory usage: {memory.percent}%")
        
        # Check disk usage
        if disk.percent > 90:
            status = "degraded"
            warnings.append(f"High disk usage: {disk.percent}%")
        
        health_data = {
            "status": status,
            "timestamp": self.last_check,
            "uptime_seconds": round(uptime_seconds, 2),
            "system": {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
            },
            "application": {
                "name": "Xponent Shopping Assistant",
                "version": "1.0.0"
            }
        }
        
        if warnings:
            health_data["warnings"] = warnings
        
        self.status = status
        return health_data


# Global health checker instance
_health_checker = None


def get_health_checker(app_instance):
    """Get or create health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(app_instance)
    return _health_checker


def health_check_endpoint() -> str:
    """
    Health check endpoint that returns JSON string
    Returns health status for Docker container health checks
    """
    global _health_checker
    
    if _health_checker is None:
        # Return basic health if checker not initialized
        return json.dumps({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "application": {
                "name": "Xponent Shopping Assistant",
                "version": "1.0.0"
            }
        })
    
    health_data = _health_checker.check_health()
    return json.dumps(health_data)


def reset_health_checker():
    """Reset the global health checker instance"""
    global _health_checker
    _health_checker = None
