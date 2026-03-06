import { Button } from "@/components/ui/button";
import { ArrowRight, Brain, Zap, Shield } from "lucide-react";
import { useNavigate } from "react-router-dom";
import heroBg from "@/assets/hero-bg.jpg";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div 
          className="absolute inset-0 z-0"
          style={{
            backgroundImage: `url(${heroBg})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            opacity: 0.3
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/50 via-background/80 to-background z-0" />
        
        <div className="relative z-10 container mx-auto px-4 py-24 md:py-32">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <div className="inline-block">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-secondary border border-primary/20 mb-6">
                <Zap className="h-4 w-4 text-accent" />
                <span className="text-sm font-medium">AI Model Generation Platform</span>
              </div>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent animate-gradient">
              ModelForge AI
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              Create custom AI models without code. From NLP to chatbots, train and deploy your models in minutes.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-[var(--glow-primary)] transition-all hover:scale-105"
                onClick={() => navigate('/create')}
              >
                Start Building
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-primary/50 hover:bg-secondary"
              >
                View Documentation
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 container mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="text-center space-y-4 p-6 rounded-lg bg-gradient-to-b from-card to-secondary border border-border">
            <div className="inline-flex p-4 rounded-full bg-primary/10">
              <Brain className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold">Smart Training</h3>
            <p className="text-muted-foreground">
              Auto-configure optimal training parameters for your specific task and dataset
            </p>
          </div>
          
          <div className="text-center space-y-4 p-6 rounded-lg bg-gradient-to-b from-card to-secondary border border-border">
            <div className="inline-flex p-4 rounded-full bg-accent/10">
              <Zap className="h-8 w-8 text-accent" />
            </div>
            <h3 className="text-xl font-semibold">Lightning Fast</h3>
            <p className="text-muted-foreground">
              Train models in minutes with our optimized infrastructure and algorithms
            </p>
          </div>
          
          <div className="text-center space-y-4 p-6 rounded-lg bg-gradient-to-b from-card to-secondary border border-border">
            <div className="inline-flex p-4 rounded-full bg-primary/10">
              <Shield className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-xl font-semibold">Enterprise Ready</h3>
            <p className="text-muted-foreground">
              Secure, scalable, and production-ready models with comprehensive testing
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
