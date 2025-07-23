#pragma once

#define GLFW_INCLUDE_VULKAN

#include <vector>
#include <string>

#include "../WindowsSecurityAttributes/WindowsSecurityAttributes.h"

#include <cuda.h>
#include <cuda_runtime.h>

// glfw
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Vulkan
#include <vulkan/vulkan.h>
#ifdef _WIN64
#include <vulkan/vulkan_win32.h>
#endif

#include "Math/Math.h"
#include "../Helper/helper_timer.h"

namespace aco
{
	namespace gfx
	{
		constexpr int WIDTH = 1920;
		constexpr int HEIGHT = 1080;
		constexpr int MAX_FRAMES = 4;

		class VulkanRenderer
		{
		public:
			void Run();

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
			void cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore);
			void cudaVkSemaphoreWait(cudaExternalSemaphore_t& extSemaphore);
			void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
			void submitVulkan(uint32_t imageIndex);
			void submitVulkanCuda(uint32_t imageIndex);

			void initWindow();
			void initVulkan();
			void initCuda();
			void initImgui();
			void cleanupSwapChain();
			void cleanup();

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
			void generateMipmaps(VkImage image, VkFormat imageFormat);
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
			void cudaVkImportSemaphore();
			void cudaVkImportImageMem();
			void cudaUpdateVkImage();
			void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
			void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

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
			VkSemaphore cudaUpdateVkSemaphore, mVkUpdateCudaSemaphore;
			std::vector<VkFence> mInFlightFences;

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

			unsigned int mMipLevels = 1;
			size_t mTotalImageMemSize;

			// CUDA objects
			cudaExternalMemory_t mCudaExtMemImageBuffer;
			cudaMipmappedArray_t mCudaMipmappedImageArray;
			cudaMipmappedArray_t mCudaMipmappedImageArrayTemp;
			cudaMipmappedArray_t mCudaMipmappedImageArrayOrig;
			std::vector<cudaSurfaceObject_t> mSurfaceObjectList;
			std::vector<cudaSurfaceObject_t> mSurfaceObjectListTemp;
			cudaSurfaceObject_t* dev_mSurfaceObjectList;
			cudaSurfaceObject_t* dev_mSurfaceObjectListTemp;
			cudaTextureObject_t mTextureObjMipMapInput;

			cudaExternalSemaphore_t mCudaExtCudaUpdateVkSemaphore;
			cudaExternalSemaphore_t mCudaExtVkUpdateCudaSemaphore;
			cudaStream_t mStreamToRun;

			StopWatchInterface* mTimer = nullptr;
			float mDeltaTime;

			// GUI supports
			bool mbMinimized = false;
		};
	}
}