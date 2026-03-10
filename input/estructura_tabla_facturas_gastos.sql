-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1:3306
-- Tiempo de generación: 15-07-2025 a las 10:49:54
-- Versión del servidor: 8.0.31
-- Versión de PHP: 8.2.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `bluebillprod`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `facturas_gastos`
--

DROP TABLE IF EXISTS `facturas_gastos`;
CREATE TABLE IF NOT EXISTS `facturas_gastos` (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `numero_factura` varchar(40) COLLATE utf8mb4_unicode_ci NOT NULL,
  `fecha_emision` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL,
  `nombre_emisor` varchar(120) COLLATE utf8mb4_unicode_ci NOT NULL,
  `cif_emisor` varchar(120) COLLATE utf8mb4_unicode_ci NOT NULL,
  `direccion_emisor` varchar(199) COLLATE utf8mb4_unicode_ci NOT NULL,
  `email_emisor` varchar(150) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `base_imponible` decimal(10,2) NOT NULL,
  `total_impuestos` decimal(10,2) NOT NULL,
  `total_irpf` decimal(10,2) NOT NULL,
  `total_factura` decimal(10,2) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Volcado de datos para la tabla `facturas_gastos`
--

INSERT INTO `facturas_gastos` (`id`, `numero_factura`, `fecha_emision`, `nombre_emisor`, `cif_emisor`, `direccion_emisor`, `email_emisor`, `base_imponible`, `total_impuestos`, `total_irpf`, `total_factura`, `created_at`, `updated_at`) VALUES
(10, '000100060', '2025-07-01 12:00:00', 'BSGSPAIN', 'ESB10691061', '200 P.º de la Castellana 28046 - Madrid (Madrid)', 'info@bsgspain.es', '1398.88', '293.77', '97.92', '1594.73', '2025-07-15 08:47:57', '2025-07-15 08:47:57'),
(11, '000100055', '2025-06-25 12:00:00', 'BSGSPAIN', 'ESB10691061', '200 P.º de la Castellana 28046 - Madrid (Madrid)', 'info@bsgspain.es', '10.99', '2.31', '0.00', '13.30', '2025-07-15 08:48:05', '2025-07-15 08:48:05'),
(12, '000100059', '2025-07-01 12:00:00', 'BSGSPAIN', 'ESB10691061', '200 P.º de la Castellana 28046 - Madrid (Madrid)', 'info@bsgspain.es', '1299.99', '273.00', '0.00', '1572.99', '2025-07-15 08:48:14', '2025-07-15 08:48:14');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
