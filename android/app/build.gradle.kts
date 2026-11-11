plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.game2048.ai"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.game2048.ai"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation("org.pytorch:pytorch_android_lite:2.2.0")
    implementation("org.pytorch:pytorch_android_torchvision_lite:2.2.0")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}
