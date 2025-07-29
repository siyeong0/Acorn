#pragma once
// glfw
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Vulkan
#include "../WindowsSecurityAttributes/WindowsSecurityAttributes.h"
#include <vulkan/vulkan.h>
#ifdef _WIN64
#include <vulkan/vulkan_win32.h>
#endif

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// NRD
#undef max
#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIImgui.h"
#include "Extensions/NRILowLatency.h"
#include "Extensions/NRIMeshShader.h"
#include "Extensions/NRIRayTracing.h"
#include "Extensions/NRIResourceAllocator.h"
#include "Extensions/NRIStreamer.h"
#include "Extensions/NRISwapChain.h"
#include "Extensions/NRIUpscaler.h"

#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIWrapperVK.h"

#include "NRD.h"

// std
#include <array>
#include <vector>
#include <string>

#include "Math/Math.h"
#include "../Helper/helper_timer.h"
#include "../CUDA/ICublit.h"
#include "../CUDA/CudaRenderBuffer.h"

namespace nrd { class Integration; }

namespace aco
{
	namespace gfx
	{
		constexpr int WIDTH = 1280;
		constexpr int HEIGHT = 720;
		constexpr int MAX_FRAMES = 4;

		class VulkanRenderer
		{
		public:
			void Run();
			void SetBlit(ICublit* cublit) { mCublit = cublit; }

		public:
			struct QueueFamilyIndices
			{
				int GraphicsFamily = -1;
				int PresentFamily = -1;
				bool IsComplete() { return GraphicsFamily >= 0 && PresentFamily >= 0; }
			};

			struct SwapChainSupportDetails
			{
				VkSurfaceCapabilitiesKHR Capabilities;
				std::vector<VkSurfaceFormatKHR> Formats;
				std::vector<VkPresentModeKHR> PresentModes;
			};

		private:
			void updateUniformBuffer();
			void drawFrame();
			void drawGui();
			void cudaVkSemaphoreWait(cudaExternalSemaphore_t& extSemaphore);
			void cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore);
			void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
			void submitVulkan(uint32_t imageIndex);
			void submitVulkanCuda(uint32_t imageIndex);

			// Initialization
			void initWindow();
			void initVulkan();
			void initCuda();
			void initNRD();
			void initImgui();
			void cleanupSwapChain();
			void cleanup();

			// Vulkan
			void createInstance();
			void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
			void setupDebugMessenger();
			void createSurface();
			void pickPhysicalDevice();
			void getKhrExtensionsFn();
			int setCudaVkDevice();
			void createLogicalDevice();
			void createSwapChain();
			void recreateSwapChain();
			void createImageViews();
			void createRenderPass();
			void createDescriptorSetLayout();
			void createGraphicsPipeline();
			void createFramebuffers();
			void createCommandPool();
			void createTextureImage();
#ifdef _WIN64 // For windows
			HANDLE getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);
			HANDLE getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore& semVkCuda);
#else
			int getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType);
			int getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore& semVkCuda);
#endif
			void createTextureImageView();
			void createTextureSampler();
			VkImageView createImageView(VkImage image, VkFormat format);
			void createImage(
				uint32_t width, uint32_t height,
				VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
				VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
			void createVertexBuffer();
			void createIndexBuffer();
			void createUniformBuffers();
			void createDescriptorPool();
			void createDescriptorSets();
			void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
			void createCommandBuffers();
			void createSyncObjects();
			void createSyncObjectsExt();

			VkCommandBuffer beginSingleTimeCommands();
			void endSingleTimeCommands(VkCommandBuffer commandBuffer);
			void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
			uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
			void createCudaRenderTarget();
			void cudaVkImportSemaphore();
			void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
			void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

			// NRD


			// Helpers
			static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
			static VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
			static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
			static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
			static VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);
			static SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
			static bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface);
			static bool checkDeviceExtensionSupport(VkPhysicalDevice device);
			static VulkanRenderer::QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
			static std::vector<const char*> getRequiredExtensions();
			static bool checkValidationLayerSupport();
			static std::vector<char> readFile(const std::string& filename);
			static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
				VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
				VkDebugUtilsMessageTypeFlagsEXT messageType,
				const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

		private:
			GLFWwindow* mWindow;

			VkInstance mInstance;
			VkDebugUtilsMessengerEXT mDebugMessenger;
			VkSurfaceKHR mSurface;

			VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
			VkDevice mDevice;
			uint8_t mVkDeviceUUID[VK_UUID_SIZE];

			VkQueue mGraphicsQueue;
			VkQueue mPresentQueue;

			VkSwapchainKHR mSwapChain;
			std::vector<VkImage> mSwapChainImages;
			VkFormat mSwapChainImageFormat;
			VkExtent2D mSwapChainExtent;
			std::vector<VkImageView> mSwapChainImageViews;
			std::vector<VkFramebuffer> mSwapChainFramebuffers;

			VkRenderPass mRenderPass;
			VkDescriptorSetLayout mDescriptorSetLayout;
			VkPipelineLayout mPipelineLayout;
			VkPipeline mGraphicsPipeline;

			VkCommandPool mCommandPool;

			VkImage mTextureImage;
			VkDeviceMemory mTextureImageMemory;
			VkImageView mTextureImageView;
			VkSampler mTextureSampler;

			VkBuffer mVertexBuffer;
			VkDeviceMemory mVertexBufferMemory;
			VkBuffer mIndexBuffer;
			VkDeviceMemory mIndexBufferMemory;

			std::vector<VkBuffer> mUniformBuffers;
			std::vector<VkDeviceMemory> mUniformBuffersMemory;

			VkDescriptorPool mDescriptorPool;
			std::vector<VkDescriptorSet> mDescriptorSets;

			std::vector<VkCommandBuffer> mCommandBuffers;

			std::vector<VkSemaphore> mImageAvailableSemaphores;
			std::vector<VkSemaphore> mRenderFinishedSemaphores;
			VkSemaphore mCudaUpdateVkSemaphore, mVkUpdateCudaSemaphore;
			std::vector<VkFence> mInFlightFences;

			size_t mImageMemSize;

			size_t mCurrentFrame = 0;

			bool mbFramebufferResized = false;

#ifdef _WIN64
			PFN_vkGetMemoryWin32HandleKHR mFpGetMemoryWin32HandleKHR;
			PFN_vkGetSemaphoreWin32HandleKHR mFpGetSemaphoreWin32HandleKHR;
#else
			PFN_vkGetMemoryFdKHR mFpGetMemoryFdKHR = NULL;
			PFN_vkGetSemaphoreFdKHR mFpGetSemaphoreFdKHR = NULL;
#endif

			PFN_vkGetPhysicalDeviceProperties2 mFpGetPhysicalDeviceProperties2;

			// CUDA objects
			ICublit* mCublit = nullptr;
			cudaExternalMemory_t mCudaExtMemImageBuffer;
			CudaRenderBuffer mCudaRenderTarget;

			cudaExternalSemaphore_t mCudaExtCudaUpdateVkSemaphore;
			cudaExternalSemaphore_t mCudaExtVkUpdateCudaSemaphore;
			cudaStream_t mStreamToRun;

			StopWatchInterface* mTimer = nullptr;
			float mDeltaTime;

			// GUI supports
			bool mbMinimized = false;

			// NRD

#define NRD_COMBINED                        1

#define NRD_MODE                            NORMAL

#define NORMAL                              0
#define SH                                  1 // NORMAL + SH (SG) resolve
#define OCCLUSION                           2
#define DIRECTIONAL_OCCLUSION               3 // diffuse OCCLUSION + SH (SG) resolve

#define SIGMA_TRANSLUCENT 1
#if (SIGMA_TRANSLUCENT == 1)
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW_TRANSLUCENCY
#else
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW
#endif

#define NRD_ID(x) nrd::Identifier(nrd::Denoiser::x)

			nrd::Integration* mNRD;
			//nrd::RelaxSettings m_RelaxSettings = {};
			//nrd::ReblurSettings m_ReblurSettings = {};
			//nrd::SigmaSettings m_SigmaSettings = {};
			//nrd::ReferenceSettings m_ReferenceSettings = {};

			// NRI
			struct NRIInterface
				: public nri::CoreInterface,
				public nri::HelperInterface,
				public nri::LowLatencyInterface,
				public nri::MeshShaderInterface,
				public nri::RayTracingInterface,
				public nri::ResourceAllocatorInterface,
				public nri::StreamerInterface,
				public nri::SwapChainInterface,
				public nri::UpscalerInterface {
				inline bool HasCore() const { return GetDeviceDesc != nullptr; }
				inline bool HasHelper() const { return CalculateAllocationNumber != nullptr; }
				inline bool HasLowLatency() const { return SetLatencySleepMode != nullptr; }
				inline bool HasMeshShader() const { return CmdDrawMeshTasks != nullptr; }
				inline bool HasRayTracing() const { return CreateRayTracingPipeline != nullptr; }
				inline bool HasResourceAllocator() const { return AllocateBuffer != nullptr; }
				inline bool HasStreamer() const { return CreateStreamer != nullptr; }
				inline bool HasSwapChain() const { return CreateSwapChain != nullptr; }
				inline bool HasUpscaler() const { return CreateUpscaler != nullptr; }
			};

			NRIInterface mNRIInterface;
			nri::Streamer* mNRIStreamer = nullptr;
			nri::Upscaler* mDLSR = nullptr;
			nri::Upscaler* mDLRR = nullptr;
			nri::Queue* mNRIGraphicsQueue = nullptr;
			nri::Fence* mNRIFrameFence = nullptr;
			std::array<nri::Upscaler*, 2> mNRIUpsaclers = {};
			
			int32_t mDlssQuality = int32_t(-1);
			nri::UpscalerType mUpscalerType = nri::UpscalerType::DLSR;
			bool mbUseDlssTnn = false;

			bool mbVsync = false;
			bool mbReversedZ = false;

			static constexpr nri::VKBindingOffsets VK_BINDING_OFFSETS = { 0, 128, 32, 64 };
			static constexpr uint32_t DYNAMIC_CONSTANT_BUFFER_SIZE = 1024 * 1024; // 1MB
			static constexpr bool NRD_ENABLE_WHOLE_LIFETIME_DESCRIPTOR_CACHING = true;
			static constexpr bool NRD_PROMOTE_FLOAT16_TO_32 = false;
			static constexpr bool NRD_DEMOTE_FLOAT32_TO_16 = false;
			static constexpr bool NRD_USE_AUTO_WRAPPER = false;
		};
	}
}