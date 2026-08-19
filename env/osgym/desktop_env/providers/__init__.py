from .base import VMManager, Provider


def create_vm_manager_and_provider(provider_name: str, region: str):
    """Create the Docker VM manager and provider used by OSGym."""
    provider_name = provider_name.lower().strip()
    if provider_name != "docker":
        raise ValueError(
            f"Unsupported OSGym provider {provider_name!r}; only 'docker' is supported"
        )

    from .docker.manager import DockerVMManager
    from .docker.provider import DockerProvider

    return DockerVMManager(), DockerProvider(region)
