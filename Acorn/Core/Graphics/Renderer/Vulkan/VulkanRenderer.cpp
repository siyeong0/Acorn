#include "VulkanRenderer.h"

#include <iostream>
#include <fstream>
#include <array>
#include <string>
#include <format>
#include <set>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "Debug.h"
#include "Math/Math.h"

namespace aco
{
	namespace gfx
	{
		struct Vertex
		{
			FVector4 Position;
			FVector3 Color;
			FVector2 TexCoord;

			static VkVertexInputBindingDescription GetBindingDescription()
			{
				VkVertexInputBindingDescription bindingDescription = {};
				bindingDescription.binding = 0;
				bindingDescription.stride = sizeof(Vertex);
				bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

				return bindingDescription;
			}

			static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions()
			{
				std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

				attributeDescriptions[0].binding = 0;
				attributeDescriptions[0].location = 0;
				attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
				attributeDescriptions[0].offset = offsetof(Vertex, Position);

				attributeDescriptions[1].binding = 0;
				attributeDescriptions[1].location = 1;
				attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
				attributeDescriptions[1].offset = offsetof(Vertex, Color);

				attributeDescriptions[2].binding = 0;
				attributeDescriptions[2].location = 2;
				attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
				attributeDescriptions[2].offset = offsetof(Vertex, TexCoord);

				return attributeDescriptions;
			}
		};

		struct UniformBufferObject
		{
			alignas(16) FMatrix4x4 Model;
			alignas(16) FMatrix4x4 View;
			alignas(16) FMatrix4x4 Proj;
		};

		const std::vector<Vertex> sVertices =
		{
			{{-1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{1.0f, -1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{1.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
			{{-1.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}} };

		const std::vector<uint16_t> sIndices = { 0, 1, 2, 2, 3, 0 };

		const std::vector<const char*> sValidationLayers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
		const bool sEnableValidationLayers = true;
#else
		const bool sEnableValidationLayers = true;
#endif

		VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
			const VkAllocationCallbacks* pAllocator,
			VkDebugUtilsMessengerEXT* pDebugMessenger)
		{
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr)
			{
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			}
			else
			{
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		};

		const std::vector<const char*> deviceExtensions =
		{
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		#ifdef _WIN64
			VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
		#else
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
		#endif
		};

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			if (key == GLFW_KEY_UP && action == GLFW_RELEASE)
			{

			}

			if (key == GLFW_KEY_DOWN && action == GLFW_RELEASE)
			{

			}
		}

		void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
			const VkAllocationCallbacks* pAllocator)
		{
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr)
			{
				func(instance, debugMessenger, pAllocator);
			}
		}

		void VulkanRenderer::Run()
		{
			initWindow();
			initVulkan();
			initCuda();
			initImgui();

			// Main loop.

			updateUniformBuffer();
			while (!glfwWindowShouldClose(mWindow))
			{
				glfwPollEvents();
				drawGui();
				if (!mbMinimized)
				{
					drawFrame();
				}
			}

			vkDeviceWaitIdle(mDevice);
			cleanup();
		}

		void VulkanRenderer::updateUniformBuffer()
		{
			UniformBufferObject ubo = {};

			// 이미지를 화면에 꽉 차게 보여주는 시점으로 설정
			ubo.Model = FMatrix4x4::Identity();
			Mat4x4Translate(ubo.Model, 0.0f, 0.0f, 0.2f);
			ubo.View = FMatrix4x4::Identity();
			FVector3 eye = { 0.0f, 0.0f, -5.0f };
			FVector3 center = { 0.0f, 0.0f, 1.0f };
			FVector3 up = { 0.0f, 1.0f, 0.0f };
			Mat4x4LookAt(ubo.View, eye, center, up);
			Mat4x4Ortho(ubo.Proj, -1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 10.0f);

			for (size_t i = 0; i < mSwapChainImages.size(); i++)
			{
				void* data;
				vkMapMemory(mDevice, mUniformBuffersMemory[i], 0, sizeof(ubo), 0, &data);
				memcpy(data, &ubo, sizeof(ubo));
				vkUnmapMemory(mDevice, mUniformBuffersMemory[i]);
			}
		}

		void VulkanRenderer::drawFrame()
		{
			sdkStartTimer(&mTimer);

			static int startSubmit = 0;

			vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

			uint32_t imageIndex;
			VkResult result =
				vkAcquireNextImageKHR(mDevice, mSwapChain, std::numeric_limits<uint64_t>::max(),
					mImageAvailableSemaphores[mCurrentFrame], VK_NULL_HANDLE, &imageIndex);

			if (result == VK_ERROR_OUT_OF_DATE_KHR)
			{
				recreateSwapChain();
				return;
			}
			else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			{
				throw std::runtime_error(std::format("Failed to acquire swap chain image!. ({})", dbg::VkResultToString(result)));
			}

			vkResetFences(mDevice, 1, &mInFlightFences[mCurrentFrame]);

			recordCommandBuffer(mCommandBuffers[imageIndex], imageIndex);
			if (!startSubmit)
			{
				submitVulkan(imageIndex);
				startSubmit = 1;
			}
			else
			{
				submitVulkanCuda(imageIndex);
			}

			VkPresentInfoKHR presentInfo = {};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			VkSemaphore signalSemaphores[] = { mRenderFinishedSemaphores[mCurrentFrame] };

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = signalSemaphores;

			VkSwapchainKHR swapChains[] = { mSwapChain };
			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = swapChains;
			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr; // Optional

			result = vkQueuePresentKHR(mPresentQueue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || mbFramebufferResized)
			{
				mbFramebufferResized = false;
				recreateSwapChain();
			}
			else if (result != VK_SUCCESS)
			{
				throw std::runtime_error(std::format("Failed to acquire swap chain image!. ({})", dbg::VkResultToString(result)));
			}

			cudaUpdateVkImage();

			sdkStopTimer(&mTimer);
			mDeltaTime = sdkGetAverageTimerValue(&mTimer) / 1000.0f;
			float avgFps = 1.0f / mDeltaTime;
			// sdkResetTimer(&timer);

			mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES;
			// Added sleep of 10 millisecs so that CPU does not submit too much work
			// to GPU
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			char title[256];
			sprintf(title, "Acorn %3.1f fps", avgFps);
			glfwSetWindowTitle(mWindow, title);
		}

		void VulkanRenderer::drawGui()
		{
			// Start the Dear ImGui frame
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			{

			}

			// Rendering
			ImGui::Render();
			ImDrawData* draw_data = ImGui::GetDrawData();

			mbMinimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
		}

		void VulkanRenderer::cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore)
		{
			cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
			memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

			extSemaphoreSignalParams.params.fence.value = 0;
			extSemaphoreSignalParams.flags = 0;
			CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemaphoreSignalParams, 1, mStreamToRun));
		}

		void VulkanRenderer::cudaVkSemaphoreWait(cudaExternalSemaphore_t& extSemaphore)
		{
			cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;

			memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));

			extSemaphoreWaitParams.params.fence.value = 0;
			extSemaphoreWaitParams.flags = 0;

			CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemaphoreWaitParams, 1, mStreamToRun));
		}

		void VulkanRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
		{
			// 1. Begin
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			// 2. Begin render pass
			VkClearValue clearColor = { 0.1f, 0.1f, 0.1f, 1.0f };
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = mRenderPass;
			renderPassInfo.framebuffer = mSwapChainFramebuffers[imageIndex];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = mSwapChainExtent;
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);

			VkBuffer vertexBuffers[] = { mVertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffer, mIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipelineLayout,
				0, 1, &mDescriptorSets[imageIndex], 0, nullptr);

			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(sIndices.size()), 1, 0, 0, 0);
			// vkCmdDraw(commandBuffers[i],
			// static_cast<uint32_t>(vertices.size()), 1, 0, 0);

			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

			vkCmdEndRenderPass(commandBuffer);
			vkEndCommandBuffer(commandBuffer);
		}

		void VulkanRenderer::submitVulkan(uint32_t imageIndex)
		{
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore waitSemaphores[] = { mImageAvailableSemaphores[mCurrentFrame] };
			VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &mCommandBuffers[imageIndex];

			VkSemaphore signalSemaphores[] =
			{
				mRenderFinishedSemaphores[mCurrentFrame],
				mVkUpdateCudaSemaphore
			};

			submitInfo.signalSemaphoreCount = 2;
			submitInfo.pSignalSemaphores = signalSemaphores;

			if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFences[mCurrentFrame]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit draw command buffer!");
			}
		}

		void VulkanRenderer::submitVulkanCuda(uint32_t imageIndex)
		{
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore waitSemaphores[] = { mImageAvailableSemaphores[mCurrentFrame], mCudaUpdateVkSemaphore };
			VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
												 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
			submitInfo.waitSemaphoreCount = 2;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &mCommandBuffers[imageIndex];

			VkSemaphore signalSemaphores[] =
			{
				mRenderFinishedSemaphores[mCurrentFrame],
				mVkUpdateCudaSemaphore
			};

			submitInfo.signalSemaphoreCount = 2;
			submitInfo.pSignalSemaphores = signalSemaphores;

			if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFences[mCurrentFrame]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to submit draw command buffer!");
			}
		}

		void VulkanRenderer::initWindow()
		{
			glfwInit();

			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

			mWindow = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Image CUDA Box Filter", nullptr, nullptr);

			glfwSetWindowUserPointer(mWindow, this);
			glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);

			glfwSetKeyCallback(mWindow, keyCallback);
		}

		void VulkanRenderer::initVulkan()
		{
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			getKhrExtensionsFn();
			createSwapChain();
			createImageViews();
			createRenderPass();
			createDescriptorSetLayout();
			createGraphicsPipeline();
			createFramebuffers();
			createCommandPool();
			createTextureImage();
			createTextureImageView();
			createTextureSampler();
			createVertexBuffer();
			createIndexBuffer();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			createCommandBuffers();
			createSyncObjects();
			createSyncObjectsExt();
		}

		void VulkanRenderer::initCuda()
		{
			setCudaVkDevice();
			CUDA_CHECK(cudaStreamCreate(&mStreamToRun));
			cudaVkImportImageMem();
			cudaVkImportSemaphore();

			sdkCreateTimer(&mTimer);
		}

		void VulkanRenderer::initImgui()
		{
			VkDescriptorPool imguiDescriptorPool;
			VkDescriptorPoolSize pool_sizes[] =
			{
				{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
				{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
				{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
				{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
				{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
				{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
			};

			VkDescriptorPoolCreateInfo pool_info = {};
			pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
			pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
			pool_info.pPoolSizes = pool_sizes;

			vkCreateDescriptorPool(mDevice, &pool_info, nullptr, &imguiDescriptorPool);

			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
			io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

			// Setup Dear ImGui style
			ImGui::StyleColorsDark();
			//ImGui::StyleColorsLight();

			// Setup Platform/Renderer backends
			ImGui_ImplGlfw_InitForVulkan(mWindow, true);
			ImGui_ImplVulkan_InitInfo init_info = {};
			//init_info.ApiVersion = VK_API_VERSION_1_3;              // Pass in your value of VkApplicationInfo::apiVersion, otherwise will default to header version.
			init_info.Instance = mInstance;
			init_info.PhysicalDevice = mPhysicalDevice;
			init_info.Device = mDevice;
			init_info.QueueFamily = findQueueFamilies(mPhysicalDevice, mSurface).GraphicsFamily;
			init_info.Queue = mGraphicsQueue;
			init_info.DescriptorPool = imguiDescriptorPool;
			init_info.RenderPass = mRenderPass;
			init_info.Subpass = 0;
			init_info.MinImageCount = 3;
			init_info.ImageCount = 3;
			init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
			init_info.Allocator = nullptr;
			ImGui_ImplVulkan_Init(&init_info);
		}

		void VulkanRenderer::cleanupSwapChain()
		{
			for (auto framebuffer : mSwapChainFramebuffers)
			{
				vkDestroyFramebuffer(mDevice, framebuffer, nullptr);
			}

			vkFreeCommandBuffers(mDevice, mCommandPool, static_cast<uint32_t>(mCommandBuffers.size()), mCommandBuffers.data());

			vkDestroyPipeline(mDevice, mGraphicsPipeline, nullptr);
			vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
			vkDestroyRenderPass(mDevice, mRenderPass, nullptr);

			for (auto imageView : mSwapChainImageViews)
			{
				vkDestroyImageView(mDevice, imageView, nullptr);
			}

			vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);

			for (size_t i = 0; i < mSwapChainImages.size(); i++)
			{
				vkDestroyBuffer(mDevice, mUniformBuffers[i], nullptr);
				vkFreeMemory(mDevice, mUniformBuffersMemory[i], nullptr);
			}

			vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);
		}

		void VulkanRenderer::cleanup()
		{
			cleanupSwapChain();

			vkDestroySampler(mDevice, mTextureSampler, nullptr);
			vkDestroyImageView(mDevice, mTextureImageView, nullptr);

			CUDA_CHECK(cudaDestroySurfaceObject(mSurfaceObject));
			CUDA_CHECK(cudaDestroySurfaceObject(mSurfaceObjectTemp));

			sdkDeleteTimer(&mTimer);

			CUDA_CHECK(cudaFree(dev_mSurfaceObject));
			CUDA_CHECK(cudaFree(dev_mSurfaceObjectTemp));
			CUDA_CHECK(cudaFreeMipmappedArray(mCudaImageArrayTemp));
			CUDA_CHECK(cudaFreeMipmappedArray(mCudaImageArrayOrig));
			CUDA_CHECK(cudaFreeMipmappedArray(mCudaImageArray));
			CUDA_CHECK(cudaDestroyTextureObject(mTextureObjInput));
			CUDA_CHECK(cudaDestroyExternalMemory(mCudaExtMemImageBuffer));
			CUDA_CHECK(cudaDestroyExternalSemaphore(mCudaExtCudaUpdateVkSemaphore));
			CUDA_CHECK(cudaDestroyExternalSemaphore(mCudaExtVkUpdateCudaSemaphore));

			vkDestroyImage(mDevice, mTextureImage, nullptr);
			vkFreeMemory(mDevice, mTextureImageMemory, nullptr);

			vkDestroyDescriptorSetLayout(mDevice, mDescriptorSetLayout, nullptr);

			vkDestroyBuffer(mDevice, mIndexBuffer, nullptr);
			vkFreeMemory(mDevice, mIndexBufferMemory, nullptr);

			vkDestroyBuffer(mDevice, mVertexBuffer, nullptr);
			vkFreeMemory(mDevice, mVertexBufferMemory, nullptr);

			vkDestroySemaphore(mDevice, mCudaUpdateVkSemaphore, nullptr);
			vkDestroySemaphore(mDevice, mVkUpdateCudaSemaphore, nullptr);

			for (size_t i = 0; i < MAX_FRAMES; i++)
			{
				vkDestroySemaphore(mDevice, mRenderFinishedSemaphores[i], nullptr);
				vkDestroySemaphore(mDevice, mImageAvailableSemaphores[i], nullptr);
				vkDestroyFence(mDevice, mInFlightFences[i], nullptr);
			}

			vkDestroyCommandPool(mDevice, mCommandPool, nullptr);

			vkDestroyDevice(mDevice, nullptr);

			if (sEnableValidationLayers)
			{
				DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
			}

			vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
			vkDestroyInstance(mInstance, nullptr);

			glfwDestroyWindow(mWindow);

			glfwTerminate();
		}

		void VulkanRenderer::createInstance()
		{
#pragma warning(disable : 6237)
			if (sEnableValidationLayers && !checkValidationLayerSupport())
#pragma warning(default : 6237)
			{
				throw std::runtime_error("validation layers requested, but not available!");
			}

			VkApplicationInfo appInfo = {};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "Vulkan Image CUDA Interop";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName = "No Engine";
			appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.apiVersion = VK_API_VERSION_1_1;

			VkInstanceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;

			auto extensions = getRequiredExtensions();
			createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			createInfo.ppEnabledExtensionNames = extensions.data();

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
			if (sEnableValidationLayers)
			{
				createInfo.enabledLayerCount = static_cast<uint32_t>(sValidationLayers.size());
				createInfo.ppEnabledLayerNames = sValidationLayers.data();

				populateDebugMessengerCreateInfo(debugCreateInfo);
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
			}
			else
			{
				createInfo.enabledLayerCount = 0;
				createInfo.pNext = nullptr;
			}

			VK_CHECK(vkCreateInstance(&createInfo, nullptr, &mInstance));

			mFpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceProperties2");
			if (mFpGetPhysicalDeviceProperties2 == NULL)
			{
				throw std::runtime_error("Vulkan: Proc address for vkGetPhysicalDeviceProperties2KHR not found.\n");
			}

#ifdef _WIN64
			mFpGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(mInstance, "vkGetMemoryWin32HandleKHR");
			if (mFpGetMemoryWin32HandleKHR == NULL)
			{
				throw std::runtime_error("Vulkan: Proc address for vkGetMemoryWin32HandleKHR not found.\n");
			}
#else
			mFpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(instance, "vkGetMemoryFdKHR");
			if (mFpGetMemoryFdKHR == NULL)
			{
				throw std::runtime_error("Vulkan: Proc address for vkGetMemoryFdKHR not found.\n");
			}
			else
			{
				std::cout << "Vulkan proc address for vkGetMemoryFdKHR - " << mFpGetMemoryFdKHR << std::endl;
			}
#endif
		}

		void VulkanRenderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
		{
			createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
		}

		void VulkanRenderer::setupDebugMessenger()
		{
			if (!sEnableValidationLayers)
			{
				return;
			}

			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);

			VK_CHECK(CreateDebugUtilsMessengerEXT(mInstance, &createInfo, nullptr, &mDebugMessenger));
		}

		void VulkanRenderer::createSurface()
		{
			VK_CHECK(glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface));
		}

		void VulkanRenderer::pickPhysicalDevice()
		{
			uint32_t deviceCount = 0;
			VK_CHECK(vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr));

			if (deviceCount == 0)
			{
				throw std::runtime_error("Failed to find GPUs with Vulkan support!");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

			for (const auto& device : devices)
			{
				if (isDeviceSuitable(device, mSurface))
				{
					mPhysicalDevice = device;
					break;
				}
			}

			if (mPhysicalDevice == VK_NULL_HANDLE)
			{
				throw std::runtime_error("Failed to find a suitable GPU!");
			}

			std::cout << "Selected physical device = " << mPhysicalDevice << std::endl;

			VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
			vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
			vkPhysicalDeviceIDProperties.pNext = NULL;

			VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
			vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

			mFpGetPhysicalDeviceProperties2(mPhysicalDevice, &vkPhysicalDeviceProperties2);

			memcpy(mVkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID, sizeof(mVkDeviceUUID));
		}

		void VulkanRenderer::getKhrExtensionsFn()
		{
#ifdef _WIN64
			mFpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(mDevice, "vkGetSemaphoreWin32HandleKHR");
			if (mFpGetSemaphoreWin32HandleKHR == NULL)
			{
				throw std::runtime_error("Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not found.\n");
			}
#else
			mFpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
			if (mFpGetSemaphoreFdKHR == NULL)
			{
				throw std::runtime_error("Vulkan: Proc address for ""\"vkGetSemaphoreFdKHR\" not found.\n");
			}
#endif
		}

		int VulkanRenderer::setCudaVkDevice()
		{
			int currentDevice = 0;
			int deviceCount = 0;
			int devicesProhibited = 0;

			cudaDeviceProp deviceProp;
			CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

			if (deviceCount == 0)
			{
				fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
				exit(EXIT_FAILURE);
			}

			// Find the GPU which is selected by Vulkan
			while (currentDevice < deviceCount)
			{
				cudaGetDeviceProperties(&deviceProp, currentDevice);

				if ((deviceProp.computeMode != cudaComputeModeProhibited))
				{
					// Compare the cuda device UUID with vulkan UUID
					int ret = memcmp(&deviceProp.uuid, &mVkDeviceUUID, VK_UUID_SIZE);
					if (ret == 0)
					{
						CUDA_CHECK(cudaSetDevice(currentDevice));
						CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, currentDevice));
						printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
							currentDevice, deviceProp.name, deviceProp.major, deviceProp.minor);

						return currentDevice;
					}

				}
				else
				{
					devicesProhibited++;
				}
				currentDevice++;
			}

			if (devicesProhibited == deviceCount)
			{
				fprintf(stderr, "CUDA error: No Vulkan-CUDA Interop capable GPU found.\n");
				exit(EXIT_FAILURE);
			}

			return -1;
		}

		void VulkanRenderer::createLogicalDevice()
		{
			QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice, mSurface);

			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
			std::set<int> uniqueQueueFamilies = { indices.GraphicsFamily, indices.PresentFamily };

			float queuePriority = 1.0f;
			for (int queueFamily : uniqueQueueFamilies)
			{
				VkDeviceQueueCreateInfo queueCreateInfo = {};
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = queueFamily;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &queuePriority;
				queueCreateInfos.push_back(queueCreateInfo);
			}

			VkPhysicalDeviceFeatures deviceFeatures = {};
			deviceFeatures.samplerAnisotropy = VK_TRUE;

			VkDeviceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

			createInfo.pQueueCreateInfos = queueCreateInfos.data();
			createInfo.queueCreateInfoCount = uint32_t(queueCreateInfos.size());

			createInfo.pEnabledFeatures = &deviceFeatures;
			std::vector<const char*> enabledExtensionNameList;

			for (int i = 0; i < deviceExtensions.size(); i++)
			{
				enabledExtensionNameList.push_back(deviceExtensions[i]);
			}
			if (sEnableValidationLayers)
			{
				createInfo.enabledLayerCount = static_cast<uint32_t>(sValidationLayers.size());
				createInfo.ppEnabledLayerNames = sValidationLayers.data();
			}
			else
			{
				createInfo.enabledLayerCount = 0;
			}
			createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNameList.size());
			createInfo.ppEnabledExtensionNames = enabledExtensionNameList.data();

			VK_CHECK(vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice));
			vkGetDeviceQueue(mDevice, indices.GraphicsFamily, 0, &mGraphicsQueue);
			vkGetDeviceQueue(mDevice, indices.PresentFamily, 0, &mPresentQueue);
		}

		void VulkanRenderer::createSwapChain()
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(mPhysicalDevice, mSurface);

			VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.Formats);
			VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.PresentModes);
			VkExtent2D extent = chooseSwapExtent(mWindow, swapChainSupport.Capabilities);

			uint32_t imageCount = swapChainSupport.Capabilities.minImageCount + 1;
			if (swapChainSupport.Capabilities.maxImageCount > 0 &&
				imageCount > swapChainSupport.Capabilities.maxImageCount)
			{
				imageCount = swapChainSupport.Capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = mSurface;

			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = surfaceFormat.format;
			createInfo.imageColorSpace = surfaceFormat.colorSpace;
			createInfo.imageExtent = extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice, mSurface);
			uint32_t queueFamilyIndices[] = { (uint32_t)indices.GraphicsFamily,
											 (uint32_t)indices.PresentFamily };

			if (indices.GraphicsFamily != indices.PresentFamily)
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			}
			else
			{
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			}

			createInfo.preTransform = swapChainSupport.Capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;

			VK_CHECK(vkCreateSwapchainKHR(mDevice, &createInfo, nullptr, &mSwapChain));

			vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, nullptr);
			mSwapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, mSwapChainImages.data());

			mSwapChainImageFormat = surfaceFormat.format;
			mSwapChainExtent = extent;
		}

		void VulkanRenderer::recreateSwapChain()
		{
			int width = 0, height = 0;
			while (width == 0 || height == 0)
			{
				glfwGetFramebufferSize(mWindow, &width, &height);
				glfwWaitEvents();
			}

			vkDeviceWaitIdle(mDevice);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createRenderPass();
			createGraphicsPipeline();
			createFramebuffers();
			createUniformBuffers();
			createDescriptorPool();
			createDescriptorSets();
			createCommandBuffers();
		}

		void VulkanRenderer::createImageViews()
		{
			mSwapChainImageViews.resize(mSwapChainImages.size());

			for (size_t i = 0; i < mSwapChainImages.size(); i++)
			{
				mSwapChainImageViews[i] = createImageView(mSwapChainImages[i], mSwapChainImageFormat);
			}
		}

		void VulkanRenderer::createRenderPass()
		{
			VkAttachmentDescription colorAttachment = {};
			colorAttachment.format = mSwapChainImageFormat;
			colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentRef = {};
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass = {};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &colorAttachmentRef;

			VkSubpassDependency dependency = {};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.pAttachments = &colorAttachment;
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies = &dependency;

			VK_CHECK(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPass));
		}

		void VulkanRenderer::createDescriptorSetLayout()
		{
			VkDescriptorSetLayoutBinding uboLayoutBinding = {};
			uboLayoutBinding.binding = 0;
			uboLayoutBinding.descriptorCount = 1;
			uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboLayoutBinding.pImmutableSamplers = nullptr;
			uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

			VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
			samplerLayoutBinding.binding = 1;
			samplerLayoutBinding.descriptorCount = 1;
			samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			samplerLayoutBinding.pImmutableSamplers = nullptr;
			samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
			VkDescriptorSetLayoutCreateInfo layoutInfo = {};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
			layoutInfo.pBindings = bindings.data();

			VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mDescriptorSetLayout));
		}

		void VulkanRenderer::createGraphicsPipeline()
		{
			auto vertShaderCode = readFile("./Shaders/vert.spv");
			auto fragShaderCode = readFile("./Shaders/frag.spv");

			VkShaderModule vertShaderModule = createShaderModule(mDevice, vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(mDevice, fragShaderCode);

			VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
			vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			vertShaderStageInfo.module = vertShaderModule;
			vertShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
			fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragShaderStageInfo.module = fragShaderModule;
			fragShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

			VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

			auto bindingDescription = Vertex::GetBindingDescription();
			auto attributeDescriptions = Vertex::GetAttributeDescriptions();

			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.vertexAttributeDescriptionCount =
				static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
			vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

			VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
			inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssembly.primitiveRestartEnable = VK_FALSE;

			VkViewport viewport = {};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)mSwapChainExtent.width;
			viewport.height = (float)mSwapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {};
			scissor.offset = { 0, 0 };
			scissor.extent = mSwapChainExtent;

			VkPipelineViewportStateCreateInfo viewportState = {};
			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = 1;
			viewportState.pViewports = &viewport;
			viewportState.scissorCount = 1;
			viewportState.pScissors = &scissor;

			VkPipelineRasterizationStateCreateInfo rasterizer = {};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE;
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterizer.depthBiasEnable = VK_FALSE;

			VkPipelineMultisampleStateCreateInfo multisampling = {};
			multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable = VK_FALSE;
			multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

			VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
				VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable = VK_FALSE;

			VkPipelineColorBlendStateCreateInfo colorBlending = {};
			colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.logicOpEnable = VK_FALSE;
			colorBlending.logicOp = VK_LOGIC_OP_COPY;
			colorBlending.attachmentCount = 1;
			colorBlending.pAttachments = &colorBlendAttachment;
			colorBlending.blendConstants[0] = 0.0f;
			colorBlending.blendConstants[1] = 0.0f;
			colorBlending.blendConstants[2] = 0.0f;
			colorBlending.blendConstants[3] = 0.0f;

			VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = 1;
			pipelineLayoutInfo.pSetLayouts = &mDescriptorSetLayout;

			VK_CHECK(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout));

			VkGraphicsPipelineCreateInfo pipelineInfo = {};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssembly;
			pipelineInfo.pViewportState = &viewportState;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState = &multisampling;
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.layout = mPipelineLayout;
			pipelineInfo.renderPass = mRenderPass;
			pipelineInfo.subpass = 0;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

			VK_CHECK(vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mGraphicsPipeline));

			vkDestroyShaderModule(mDevice, fragShaderModule, nullptr);
			vkDestroyShaderModule(mDevice, vertShaderModule, nullptr);
		}

		void VulkanRenderer::createFramebuffers()
		{
			mSwapChainFramebuffers.resize(mSwapChainImageViews.size());

			for (size_t i = 0; i < mSwapChainImageViews.size(); i++)
			{
				VkImageView attachments[] = { mSwapChainImageViews[i] };

				VkFramebufferCreateInfo framebufferInfo = {};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = mRenderPass;
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments = attachments;
				framebufferInfo.width = mSwapChainExtent.width;
				framebufferInfo.height = mSwapChainExtent.height;
				framebufferInfo.layers = 1;

				VK_CHECK(vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mSwapChainFramebuffers[i]));
			}
		}

		void VulkanRenderer::createCommandPool()
		{
			QueueFamilyIndices queueFamilyIndices = findQueueFamilies(mPhysicalDevice, mSurface);

			VkCommandPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.queueFamilyIndex = queueFamilyIndices.GraphicsFamily;

			VK_CHECK(vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool));
		}

		void VulkanRenderer::createTextureImage()
		{
			VkDeviceSize imageSize = WIDTH * HEIGHT * 4;

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;
			createBuffer(imageSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

			//void* data;
			//vkMapMemory(mDevice, stagingBufferMemory, 0, imageSize, 0, &data);
			//memcpy(data, mImageData.data(), static_cast<size_t>(imageSize));
			//vkUnmapMemory(mDevice, stagingBufferMemory);

			// VK_FORMAT_R8G8B8A8_UNORM changed to VK_FORMAT_R8G8B8A8_UINT
			createImage(WIDTH, HEIGHT,
				VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mTextureImage, mTextureImageMemory);

			transitionImageLayout(mTextureImage,
				VK_FORMAT_R8G8B8A8_UINT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			copyBufferToImage(stagingBuffer, mTextureImage,
				static_cast<uint32_t>(WIDTH),
				static_cast<uint32_t>(HEIGHT));

			vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
			vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
		}

#ifdef _WIN64 // For windows
		HANDLE VulkanRenderer::getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType)
		{
			HANDLE handle;

			VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
			vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
			vkMemoryGetWin32HandleInfoKHR.memory = mTextureImageMemory;
			vkMemoryGetWin32HandleInfoKHR.handleType = (VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

			mFpGetMemoryWin32HandleKHR(mDevice, &vkMemoryGetWin32HandleInfoKHR, &handle);
			return handle;
		}

		HANDLE VulkanRenderer::getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore& semVkCuda)
		{
			HANDLE handle;

			VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
			vulkanSemaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
			vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
			vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = semVkCuda;
			vulkanSemaphoreGetWin32HandleInfoKHR.handleType = externalSemaphoreHandleType;

			mFpGetSemaphoreWin32HandleKHR(mDevice, &vulkanSemaphoreGetWin32HandleInfoKHR, &handle);

			return handle;
		}
#else
		int VulkanRenderer::getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType)
		{
			if (externalMemoryHandleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR)
			{
				int fd;

				VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
				vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
				vkMemoryGetFdInfoKHR.pNext = NULL;
				vkMemoryGetFdInfoKHR.memory = textureImageMemory;
				vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

				mFpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd);

				return fd;
			}
			return -1;
		}

		int VulkanRenderer::getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore& semVkCuda)
		{
			if (externalSemaphoreHandleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
			{
				int fd;

				VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR = {};
				vulkanSemaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
				vulkanSemaphoreGetFdInfoKHR.pNext = NULL;
				vulkanSemaphoreGetFdInfoKHR.semaphore = semVkCuda;
				vulkanSemaphoreGetFdInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

				mFpGetSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);

				return fd;
			}
			return -1;
		}
#endif

		void VulkanRenderer::createTextureImageView()
		{
			mTextureImageView = createImageView(mTextureImage, VK_FORMAT_R8G8B8A8_UNORM);
		}

		void VulkanRenderer::createTextureSampler()
		{
			VkSamplerCreateInfo samplerInfo = {};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;
			samplerInfo.maxAnisotropy = 16;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.minLod = 0; // Optional
			samplerInfo.maxLod = static_cast<float>(1);
			samplerInfo.mipLodBias = 0; // Optional

			VK_CHECK(vkCreateSampler(mDevice, &samplerInfo, nullptr, &mTextureSampler));
		}

		VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format)
		{
			VkImageViewCreateInfo viewInfo = {};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			VkImageView imageView;
			VK_CHECK(vkCreateImageView(mDevice, &viewInfo, nullptr, &imageView));

			return imageView;
		}

		void VulkanRenderer::createImage(uint32_t width, uint32_t height, VkFormat format,
			VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
			VkImage& image, VkDeviceMemory& imageMemory)
		{
			VkImageCreateInfo imageInfo = {};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = width;
			imageInfo.extent.height = height;
			imageInfo.extent.depth = 1;
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.format = format;
			imageInfo.tiling = tiling;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = usage;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
			vkExternalMemImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
			vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
			vkExternalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
			vkExternalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

			imageInfo.pNext = &vkExternalMemImageCreateInfo;

			VK_CHECK(vkCreateImage(mDevice, &imageInfo, nullptr, &image));

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(mDevice, image, &memRequirements);

#ifdef _WIN64
			WindowsSecurityAttributes winSecurityAttributes;

			VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
			vulkanExportMemoryWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
			vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
			vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
			vulkanExportMemoryWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
			vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
			VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
			vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
			vulkanExportMemoryAllocateInfoKHR.pNext = IsWindows8OrGreater() ? &vulkanExportMemoryWin32HandleInfoKHR : NULL;
			vulkanExportMemoryAllocateInfoKHR.handleTypes =
				IsWindows8OrGreater()
				? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
				: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
			vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
			vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			VkMemoryRequirements vkMemoryRequirements = {};
			vkGetImageMemoryRequirements(mDevice, image, &vkMemoryRequirements);
			mImageMemSize = vkMemoryRequirements.size;

			VK_CHECK(vkAllocateMemory(mDevice, &allocInfo, nullptr, &mTextureImageMemory));

			vkBindImageMemory(mDevice, image, mTextureImageMemory, 0);
		}

		void VulkanRenderer::createVertexBuffer()
		{
			VkDeviceSize bufferSize = sizeof(sVertices[0]) * sVertices.size();

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;
			createBuffer(bufferSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, sVertices.data(), (size_t)bufferSize);
			vkUnmapMemory(mDevice, stagingBufferMemory);

			createBuffer(bufferSize,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT |
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				mVertexBuffer, mVertexBufferMemory);

			copyBuffer(stagingBuffer, mVertexBuffer, bufferSize);

			vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
			vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
		}

		void VulkanRenderer::createIndexBuffer()
		{
			VkDeviceSize bufferSize = sizeof(sIndices[0]) * sIndices.size();

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;
			createBuffer(bufferSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
			memcpy(data, sIndices.data(), (size_t)bufferSize);
			vkUnmapMemory(mDevice, stagingBufferMemory);

			createBuffer(bufferSize,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT |
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				mIndexBuffer, mIndexBufferMemory);

			copyBuffer(stagingBuffer, mIndexBuffer, bufferSize);

			vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
			vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
		}

		void VulkanRenderer::createUniformBuffers() {
			VkDeviceSize bufferSize = sizeof(UniformBufferObject);

			mUniformBuffers.resize(mSwapChainImages.size());
			mUniformBuffersMemory.resize(mSwapChainImages.size());

			for (size_t i = 0; i < mSwapChainImages.size(); i++)
			{
				createBuffer(bufferSize,
					VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
					VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					mUniformBuffers[i], mUniformBuffersMemory[i]);
			}
		}

		void VulkanRenderer::createDescriptorPool()
		{
			std::array<VkDescriptorPoolSize, 2> poolSizes = {};
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(mSwapChainImages.size());
			poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[1].descriptorCount = static_cast<uint32_t>(mSwapChainImages.size());

			VkDescriptorPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			poolInfo.pPoolSizes = poolSizes.data();
			poolInfo.maxSets = static_cast<uint32_t>(mSwapChainImages.size());

			VK_CHECK(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool));
		}

		void VulkanRenderer::createDescriptorSets()
		{
			std::vector<VkDescriptorSetLayout> layouts(mSwapChainImages.size(), mDescriptorSetLayout);
			VkDescriptorSetAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = mDescriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(mSwapChainImages.size());
			allocInfo.pSetLayouts = layouts.data();

			mDescriptorSets.resize(mSwapChainImages.size());
			VK_CHECK(vkAllocateDescriptorSets(mDevice, &allocInfo, mDescriptorSets.data()));

			for (size_t i = 0; i < mSwapChainImages.size(); i++)
			{
				VkDescriptorBufferInfo bufferInfo = {};
				bufferInfo.buffer = mUniformBuffers[i];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(UniformBufferObject);

				VkDescriptorImageInfo imageInfo = {};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = mTextureImageView;
				imageInfo.sampler = mTextureSampler;

				std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = mDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &bufferInfo;

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = mDescriptorSets[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &imageInfo;

				vkUpdateDescriptorSets(mDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}

		void VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
			VkMemoryPropertyFlags properties, VkBuffer& buffer,
			VkDeviceMemory& bufferMemory) {
			VkBufferCreateInfo bufferInfo = {};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VK_CHECK(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buffer));

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(mDevice, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

			VK_CHECK(vkAllocateMemory(mDevice, &allocInfo, nullptr, &bufferMemory));

			vkBindBufferMemory(mDevice, buffer, bufferMemory, 0);
		}

		void VulkanRenderer::createCommandBuffers()
		{
			mCommandBuffers.resize(mSwapChainFramebuffers.size());

			VkCommandBufferAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = mCommandPool;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = (uint32_t)mCommandBuffers.size();

			VK_CHECK(vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffers.data()));
		}

		void VulkanRenderer::createSyncObjects()
		{
			mImageAvailableSemaphores.resize(MAX_FRAMES);
			mRenderFinishedSemaphores.resize(MAX_FRAMES);
			mInFlightFences.resize(MAX_FRAMES);

			VkSemaphoreCreateInfo semaphoreInfo = {};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo = {};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			for (size_t i = 0; i < MAX_FRAMES; i++)
			{
				VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mImageAvailableSemaphores[i]));
				VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mRenderFinishedSemaphores[i]));
				VK_CHECK(vkCreateFence(mDevice, &fenceInfo, nullptr, &mInFlightFences[i]));
			}
		}

		void VulkanRenderer::createSyncObjectsExt()
		{
			VkSemaphoreCreateInfo semaphoreInfo = {};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

#ifdef _WIN64
			WindowsSecurityAttributes winSecurityAttributes;

			VkExportSemaphoreWin32HandleInfoKHR vulkanExportSemaphoreWin32HandleInfoKHR = {};
			vulkanExportSemaphoreWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
			vulkanExportSemaphoreWin32HandleInfoKHR.pNext = NULL;
			vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
			vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess =
				DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
			vulkanExportSemaphoreWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
			VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
			vulkanExportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef _WIN64
			vulkanExportSemaphoreCreateInfo.pNext =
				IsWindows8OrGreater() ? &vulkanExportSemaphoreWin32HandleInfoKHR : NULL;
			vulkanExportSemaphoreCreateInfo.handleTypes =
				IsWindows8OrGreater()
				? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
				: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
			vulkanExportSemaphoreCreateInfo.pNext = NULL;
			vulkanExportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
			semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

			VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mCudaUpdateVkSemaphore));
			VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mVkUpdateCudaSemaphore));
		}

		VkCommandBuffer VulkanRenderer::beginSingleTimeCommands()
		{
			VkCommandBufferAllocateInfo allocInfo = {};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = mCommandPool;
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(mDevice, &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			return commandBuffer;
		}

		void VulkanRenderer::endSingleTimeCommands(VkCommandBuffer commandBuffer)
		{
			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(mGraphicsQueue);

			vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
		}

		void VulkanRenderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkBufferCopy copyRegion = {};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

			endSingleTimeCommands(commandBuffer);
		}

		uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProperties);

			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			{

				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type!");
		}

		void VulkanRenderer::cudaVkImportSemaphore()
		{
			cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
			memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
			externalSemaphoreHandleDesc.type =
				IsWindows8OrGreater()
				? cudaExternalSemaphoreHandleTypeOpaqueWin32
				: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
			externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
				IsWindows8OrGreater()
				? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
				: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
				mCudaUpdateVkSemaphore);
#else
			externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
			externalSemaphoreHandleDesc.handle.fd = GetVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, cudaUpdateVkSemaphore);
#endif
			externalSemaphoreHandleDesc.flags = 0;

			CUDA_CHECK(cudaImportExternalSemaphore(&mCudaExtCudaUpdateVkSemaphore, &externalSemaphoreHandleDesc));

			memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
			externalSemaphoreHandleDesc.type = IsWindows8OrGreater()
				? cudaExternalSemaphoreHandleTypeOpaqueWin32
				: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
			externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
				IsWindows8OrGreater()
				? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
				: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
				mVkUpdateCudaSemaphore);
#else
			externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
			externalSemaphoreHandleDesc.handle.fd = GetVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, vkUpdateCudaSemaphore);
#endif
			externalSemaphoreHandleDesc.flags = 0;
			CUDA_CHECK(cudaImportExternalSemaphore(&mCudaExtVkUpdateCudaSemaphore, &externalSemaphoreHandleDesc));
			printf("CUDA Imported Vulkan semaphore\n");
		}

		void VulkanRenderer::cudaVkImportImageMem()
		{
			cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
			memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
			cudaExtMemHandleDesc.type =
				IsWindows8OrGreater()
				? cudaExternalMemoryHandleTypeOpaqueWin32
				: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
			cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
				IsWindows8OrGreater()
				? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
				: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
#else
			cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
			cudaExtMemHandleDesc.handle.fd = GetVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
			cudaExtMemHandleDesc.size = mImageMemSize;

			CUDA_CHECK(cudaImportExternalMemory(&mCudaExtMemImageBuffer, &cudaExtMemHandleDesc));

			cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;

			memset(&externalMemoryMipmappedArrayDesc, 0, sizeof(externalMemoryMipmappedArrayDesc));

			cudaExtent extent = make_cudaExtent(WIDTH, HEIGHT, 0);
			cudaChannelFormatDesc formatDesc;
			formatDesc.x = 8;
			formatDesc.y = 8;
			formatDesc.z = 8;
			formatDesc.w = 8;
			formatDesc.f = cudaChannelFormatKindUnsigned;

			externalMemoryMipmappedArrayDesc.offset = 0;
			externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
			externalMemoryMipmappedArrayDesc.extent = extent;
			externalMemoryMipmappedArrayDesc.flags = 0;
			externalMemoryMipmappedArrayDesc.numLevels = 1;

			CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mCudaImageArray, mCudaExtMemImageBuffer, &externalMemoryMipmappedArrayDesc));

			CUDA_CHECK(cudaMallocMipmappedArray(&mCudaImageArrayTemp, &formatDesc, extent, 1));
			CUDA_CHECK(cudaMallocMipmappedArray(&mCudaImageArrayOrig, &formatDesc, extent, 1));

			{
				cudaArray_t cudaArray, cudaArrayTemp, cudaArrayOrig;
				cudaResourceDesc resourceDesc;

				CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaArray, mCudaImageArray, 0));
				CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaArrayTemp, mCudaImageArrayTemp, 0));
				CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaArrayOrig, mCudaImageArrayOrig, 0));

				CUDA_CHECK(cudaMemcpy2DArrayToArray(
					cudaArrayOrig, 0, 0,
					cudaArray, 0, 0, WIDTH * sizeof(uchar4), HEIGHT,
					cudaMemcpyDeviceToDevice));

				memset(&resourceDesc, 0, sizeof(resourceDesc));
				resourceDesc.resType = cudaResourceTypeArray;
				resourceDesc.res.array.array = cudaArray;
				CUDA_CHECK(cudaCreateSurfaceObject(&mSurfaceObject, &resourceDesc));

				memset(&resourceDesc, 0, sizeof(resourceDesc));
				resourceDesc.resType = cudaResourceTypeArray;
				resourceDesc.res.array.array = cudaArrayTemp;
				CUDA_CHECK(cudaCreateSurfaceObject(&mSurfaceObjectTemp, &resourceDesc));
			}

			cudaResourceDesc resDescr;
			memset(&resDescr, 0, sizeof(cudaResourceDesc));

			resDescr.resType = cudaResourceTypeMipmappedArray;
			resDescr.res.mipmap.mipmap = mCudaImageArrayOrig;

			cudaTextureDesc texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));

			texDescr.normalizedCoords = true;
			texDescr.filterMode = cudaFilterModeLinear;
			texDescr.mipmapFilterMode = cudaFilterModeLinear;
			texDescr.addressMode[0] = cudaAddressModeWrap;
			texDescr.addressMode[1] = cudaAddressModeWrap;
			texDescr.maxMipmapLevelClamp = 0;
			texDescr.readMode = cudaReadModeNormalizedFloat;

			CUDA_CHECK(cudaCreateTextureObject(&mTextureObjInput, &resDescr, &texDescr, NULL));

			CUDA_CHECK(cudaMalloc((void**)&dev_mSurfaceObject, sizeof(cudaSurfaceObject_t)));
			CUDA_CHECK(cudaMalloc((void**)&dev_mSurfaceObjectTemp, sizeof(cudaSurfaceObject_t)));

			CUDA_CHECK(cudaMemcpy(dev_mSurfaceObject, &mSurfaceObject, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(dev_mSurfaceObjectTemp, &mSurfaceObjectTemp, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice));

			printf("CUDA Kernel Vulkan image buffer\n");
		}

		void VulkanRenderer::transitionImageLayout(
			VkImage image, VkFormat format,
			VkImageLayout oldLayout, VkImageLayout newLayout)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
				newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
			{
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
				newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
			{
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else
			{
				throw std::invalid_argument("unsupported layout transition!");
			}

			vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			endSingleTimeCommands(commandBuffer);
		}

		void VulkanRenderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
		{
			VkCommandBuffer commandBuffer = beginSingleTimeCommands();

			VkBufferImageCopy region = {};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = { width, height, 1 };

			vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

			endSingleTimeCommands(commandBuffer);
		}

		void VulkanRenderer::framebufferResizeCallback(GLFWwindow* window, int width, int height)
		{
			auto app = reinterpret_cast<VulkanRenderer*>(glfwGetWindowUserPointer(window));
			app->mbFramebufferResized = true;
		}

		VkShaderModule VulkanRenderer::createShaderModule(VkDevice device, const std::vector<char>& code)
		{
			VkShaderModuleCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = code.size();
			createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

			VkShaderModule shaderModule;
			VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

			return shaderModule;
		}

		VkSurfaceFormatKHR VulkanRenderer::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
		{
			if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
			{
				return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
			}

			for (const auto& availableFormat : availableFormats)
			{
				if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
					availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				{
					return availableFormat;
				}
			}

			return availableFormats[0];
		}

		VkPresentModeKHR VulkanRenderer::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
		{
			VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

			for (const auto& availablePresentMode : availablePresentModes)
			{
				if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					return availablePresentMode;
				}
				else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
				{
					bestMode = availablePresentMode;
				}
			}

			return bestMode;
		}

		VkExtent2D VulkanRenderer::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
		{
			if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			{
				return capabilities.currentExtent;
			}
			else
			{
				int width, height;
				glfwGetFramebufferSize(window, &width, &height);

				VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

				actualExtent.width =
					std::max(capabilities.minImageExtent.width,
						std::min(capabilities.maxImageExtent.width, actualExtent.width));
				actualExtent.height =
					std::max(capabilities.minImageExtent.height,
						std::min(capabilities.maxImageExtent.height, actualExtent.height));

				return actualExtent;
			}
		}

		VulkanRenderer::SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
		{
			SwapChainSupportDetails details;

			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.Capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

			if (formatCount != 0)
			{
				details.Formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.Formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

			if (presentModeCount != 0)
			{
				details.PresentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.PresentModes.data());
			}

			return details;
		}

		bool VulkanRenderer::isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface)
		{
			QueueFamilyIndices indices = findQueueFamilies(device, surface);

			bool extensionsSupported = checkDeviceExtensionSupport(device);

			bool swapChainAdequate = false;
			if (extensionsSupported)
			{
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
				swapChainAdequate = !swapChainSupport.Formats.empty() && !swapChainSupport.PresentModes.empty();
			}

			VkPhysicalDeviceFeatures supportedFeatures;
			vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

			return indices.IsComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
		}

		bool VulkanRenderer::checkDeviceExtensionSupport(VkPhysicalDevice device)
		{
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto& extension : availableExtensions)
			{
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
		}

		VulkanRenderer::QueueFamilyIndices VulkanRenderer::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
		{
			VulkanRenderer::QueueFamilyIndices indices;

			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

			int i = 0;
			for (const auto& queueFamily : queueFamilies)
			{
				if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					indices.GraphicsFamily = i;
				}

				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

				if (queueFamily.queueCount > 0 && presentSupport)
				{
					indices.PresentFamily = i;
				}

				if (indices.IsComplete())
				{
					break;
				}

				i++;
			}

			return indices;
		}

		std::vector<const char*> VulkanRenderer::getRequiredExtensions()
		{
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

			std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

			if (sEnableValidationLayers)
			{
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extensions;
		}

		bool VulkanRenderer::checkValidationLayerSupport()
		{
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

			for (const char* layerName : sValidationLayers)
			{
				bool layerFound = false;

				for (const auto& layerProperties : availableLayers)
				{
					if (strcmp(layerName, layerProperties.layerName) == 0)
					{
						layerFound = true;
						break;
					}
				}

				if (!layerFound)
				{
					return false;
				}
			}

			return true;
		}

		std::vector<char> VulkanRenderer::readFile(const std::string& filename)
		{

			std::ifstream file(filename, std::ios::ate | std::ios::binary);

			if (!file.is_open())
			{
				throw std::runtime_error("failed to open file!");
			}

			size_t fileSize = (size_t)file.tellg();
			std::vector<char> buffer(fileSize);

			file.seekg(0);
			file.read(buffer.data(), fileSize);

			file.close();

			return buffer;
		}

		VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData)
		{
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
		}
	}
}