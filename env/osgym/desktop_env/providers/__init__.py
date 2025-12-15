from desktop_env.providers.base import VMManager, Provider


def create_vm_manager_and_provider(provider_name: str, region: str, use_proxy: bool = False):
    """
    Factory function to get the Virtual Machine Manager and Provider instances based on the provided provider name.

    Args:
        provider_name (str): The name of the provider (e.g., "docker", "vmware")
        region (str): The region for the provider
        use_proxy (bool): Whether to use proxy-enabled providers (for future use)
    """
    provider_name = provider_name.lower().strip()
    if provider_name == "vmware":
        from desktop_env.providers.vmware.manager import VMwareVMManager
        from desktop_env.providers.vmware.provider import VMwareProvider
        return VMwareVMManager(), VMwareProvider(region)
    elif provider_name == "docker":
        from desktop_env.providers.docker.manager import DockerVMManager
        from desktop_env.providers.docker.provider import DockerProvider
        return DockerVMManager(), DockerProvider(region)
    else:
        raise NotImplementedError(f"{provider_name} not implemented! Only 'docker' and 'vmware' are supported.")
