"""
Script para generar código Python desde embeddings.proto.

Los archivos generados se guardan en la carpeta proto/:
- embeddings_pb2.py      (messages)
- embeddings_pb2_grpc.py (stubs/servicer base)

Uso:
    cd embedding_service
    python scripts/generate_grpc.py
"""
import os
import subprocess
import sys
from pathlib import Path


def generate_grpc_code():
    """Genera código Python desde embeddings.proto."""

    # Rutas
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    proto_file = project_root / "proto" / "embeddings.proto"
    output_dir = project_root / "proto"

    print("=" * 70)
    print("🔧 GENERANDO CÓDIGO gRPC")
    print("=" * 70)
    print(f"Proto file : {proto_file}")
    print(f"Output dir : {output_dir}")
    print()

    # Verificar que existe el .proto
    if not proto_file.exists():
        print(f"❌ Error: No se encontró {proto_file}")
        print()
        print("Asegúrate de que proto/embeddings.proto existe.")
        sys.exit(1)

    # Comando para generar código
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={output_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(proto_file)
    ]

    print("Ejecutando comando:")
    print(" ".join(str(c) for c in command))
    print()

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        # Verificar archivos generados
        pb2_file = output_dir / "embeddings_pb2.py"
        pb2_grpc_file = output_dir / "embeddings_pb2_grpc.py"

        if pb2_file.exists() and pb2_grpc_file.exists():
            print("=" * 70)
            print("✅ CÓDIGO gRPC GENERADO CORRECTAMENTE")
            print("=" * 70)
            print(f"✅ proto/{pb2_file.name}")
            print(f"✅ proto/{pb2_grpc_file.name}")
            print()
            print("Ahora puedes ejecutar el servidor gRPC:")
            print("   python grpc_server.py")
            print()
        else:
            print("⚠️  Archivos generados no encontrados en:")
            print(f"   {output_dir}")

    except subprocess.CalledProcessError as e:
        print("=" * 70)
        print("❌ ERROR AL GENERAR CÓDIGO")
        print("=" * 70)
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_grpc_code()